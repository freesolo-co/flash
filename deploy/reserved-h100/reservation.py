"""Bounded Lambda acquisition and one-week retention for the owned H100 fleet."""
import json
import os
import time
import uuid
from pathlib import Path

import httpx
import modal

CONFIG = json.loads(Path(__file__).with_name('config.json').read_text())
CAMPAIGN = CONFIG['campaign']
app = modal.App(CAMPAIGN)
image = modal.Image.debian_slim(python_version='3.12').pip_install('httpx==0.28.1').add_local_file(str(Path(__file__).with_name('config.json')), '/root/config.json')
ledger = modal.Dict.from_name(CAMPAIGN, create_if_missing=True)


class APIError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code
        super().__init__(f'Lambda HTTP {status}: {code}')


class API:
    def __init__(self):
        self.client = httpx.Client(base_url='https://cloud.lambda.com/api/v1/', headers={'Authorization': 'Bearer '+os.environ['LAMBDA_API_KEY']}, timeout=25)
        self.last = 0

    def call(self, method, path, body=None):
        time.sleep(max(0, self.last + 1.25 - time.monotonic()))
        self.last = time.monotonic()
        response = self.client.request(method, path, json=body)
        if response.status_code >= 300:
            try:
                code = str(response.json().get('error', {}).get('code', 'unknown'))[:100]
            except ValueError:
                code = 'non-json-response'
            raise APIError(response.status_code, code)
        return response.json()['data']


def fresh_state():
    return {'campaign': CAMPAIGN, 'mode': 'reserve_only', 'started_at': CONFIG['started_at'], 'acquire_until': CONFIG['acquire_until'], 'hold_until': CONFIG['absolute_hold_until'], 'slots': [{'number': i, 'name': f'{CAMPAIGN}-{i}', 'state': 'empty'} for i in range(1, 5)], 'notifications': []}


def owned(instance):
    tags = {x['key']: x['value'] for x in instance.get('tags', [])}
    return tags.get('reservation-campaign') == CAMPAIGN and instance.get('name') in {f'{CAMPAIGN}-{i}' for i in range(1, 5)}


def choices(catalog):
    found = []
    for rank, kind in enumerate(CONFIG['allowed_types']):
        offer = catalog.get(kind, {})
        spec = offer.get('instance_type', {})
        if spec.get('specs', {}).get('gpus') != 1 or 'H100' not in spec.get('gpu_description', ''):
            continue
        for region in offer.get('regions_with_capacity_available', []):
            if region['name'].startswith('us-'):
                found.append((rank, spec['price_cents_per_hour'], kind, region['name']))
    return sorted(found)


def firewall(api, region):
    name = CAMPAIGN+'-'+region
    rules = [{'protocol': 'tcp', 'port_range': [22,22], 'source_network': CONFIG['ssh_source_cidr'], 'description': 'Reservation operator SSH only; no inference ingress'}]
    existing = [r for r in api.call('GET', 'firewall-rulesets') if r['name']==name and r['region']['name']==region]
    if existing:
        if len(existing)!=1 or existing[0]['rules'] != rules:
            raise RuntimeError('Reservation firewall mismatch')
        return existing[0]['id']
    return api.call('POST', 'firewall-rulesets', {'name': name, 'region': region, 'rules': rules})['id']


def summary(state):
    result = {k:v for k,v in state.items() if k!='last_catalog'}
    result['active_count'] = sum(s['state']=='active' for s in state['slots'])
    result['allocated_count'] = sum(bool(s.get('instance_id')) and s['state']!='terminated' for s in state['slots'])
    return result


def step(api, state, save, now=None):
    now = time.time() if now is None else now
    if state['mode'] in {'adopted', 'closed'}:
        return state
    instances = api.call('GET', 'instances')
    mine = [x for x in instances if owned(x)]
    if len([x for x in mine if x['status']!='terminated'])>4:
        raise RuntimeError('Ownership count exceeds four; refusing acquisition')
    for slot in state['slots']:
        matches = [x for x in mine if x['name']==slot['name'] and x['status']!='terminated']
        if len(matches)>1:
            raise RuntimeError('Duplicate slot; refusing acquisition')
        if matches:
            x=matches[0]
            spec=x.get('instance_type', {})
            if spec.get('name') not in CONFIG['allowed_types'] or spec.get('specs',{}).get('gpus')!=1:
                raise RuntimeError('Unexpected reserved hardware')
            slot.update(instance_id=x['id'], state=x['status'], ip=x.get('ip'), instance_type=spec.get('name'), hourly_usd=spec.get('price_cents_per_hour',0)/100, region=x.get('region',{}).get('name'))
        elif slot.get('instance_id') and slot['state'] not in {'terminated','empty'}:
            slot['state']='missing_requires_review'
    state['checked_at']=now
    ready=sum(s['state']=='active' for s in state['slots'])
    allocated=sum(bool(s.get('instance_id')) and s['state']!='terminated' for s in state['slots'])
    state.update(active_count=ready, allocated_count=allocated)
    if ready==4 and not state.get('secured_at'):
        state.update(secured_at=now, hold_until=min(now+7*86400, CONFIG['absolute_hold_until']))
        state['notifications'].append({'event':'four_h100s_active','at':now,'delivered':False})
    save(state)
    if now>=state['hold_until']:
        ids=[x['id'] for x in mine if x['status'] not in {'terminated','terminating'}]
        if ids:
            api.call('POST','instance-operations/terminate',{'instance_ids':ids})
            state['phase']='expiring'
        elif not any(x['status']!='terminated' for x in mine):
            state['mode']='closed';state['phase']='expired'
        save(state)
        return state
    if ready==4:
        state['phase']='serving_flash_9b_base' if state['mode']=='serving' else 'held_awaiting_user_inference_instruction';save(state);return state
    if now>=state['acquire_until']:
        state['phase']='acquisition_window_ended'
        if not state.get('deadline_notified'):
            state['notifications'].append({'event':'acquisition_window_ended','at':now,'delivered':False})
            state['deadline_notified']=True
        save(state);return state
    state['phase']='acquiring'
    # One submission per tick, at most four named slots. Unknown outcomes are never retried.
    slot=next((s for s in state['slots'] if s['state']=='empty'),None)
    if slot is None:
        save(state);return state
    if now-max((s.get('submitted_at',0) for s in state['slots']),default=0)<13:
        save(state);return state
    catalog=api.call('GET','instance-types')
    available=choices(catalog)
    state['last_catalog']=[{'type':c[2],'region':c[3],'hourly_usd':c[1]/100} for c in available]
    if not available:
        save(state);return state
    _,price,kind,region=available[0]
    fw=firewall(api,region)
    marker=json.dumps({'campaign':CAMPAIGN,'purpose':'reserved for future Flash Qwen/Qwen3.5-9B-Base inference','inference_enabled':False})
    body={'region_name':region,'instance_type_name':kind,'quantity':1,'ssh_key_names':[CONFIG['ssh_key_name']],'firewall_rulesets':[{'id':fw}], 'name':slot['name'],'image':{'family':'gpu-base-24-04'},'tags':[{'key':'reservation-campaign','value':CAMPAIGN},{'key':'purpose','value':'hold-for-flash-9b-base'}], 'user_data':'#cloud-config\nwrite_files:\n  - path: /var/lib/flash-h100-reservation.json\n    permissions: "0644"\n    content: '+json.dumps(marker)+'\n'}
    slot.update(state='submission_unknown',submitted_at=now,instance_type=kind,region=region,quoted_hourly_usd=price/100,firewall_id=fw)
    save(state)  # Durable before POST; crash/retry must never submit the same slot twice.
    try:
        result=api.call('POST','instance-operations/launch',body)
    except APIError as e:
        slot['error']=str(e)
        if 400<=e.status<500 and e.status!=408:
            slot['state']='empty'  # Explicit rejection, not an ambiguous accepted request.
        save(state);return state
    except Exception as e:
        slot['error']=type(e).__name__;save(state);return state
    ids=result.get('instance_ids',[])
    if len(ids)!=1:
        slot['error']='unexpected_launch_response';save(state);return state
    slot.update(instance_id=ids[0],state='booting',accepted_at=now)
    save(state)
    return state


@app.function(image=image,secrets=[modal.Secret.from_name(CAMPAIGN)],schedule=modal.Period(minutes=1),timeout=180,max_containers=1,retries=0)
def tick():
    token=uuid.uuid4().hex
    if not ledger.put('tick_lock', {'token':token,'at':time.time()}, skip_if_exists=True):
        return summary(ledger.get('state') or fresh_state())
    state=ledger.get('state') or fresh_state()
    api=API()
    try:
        result=step(api,state,lambda s:ledger.put('state',s))
        print(json.dumps({'phase':result.get('phase'),'active':result.get('active_count',0),'slots':[s['state'] for s in result['slots']]}))
        return summary(result)
    except Exception as e:
        state['last_error']={'type':type(e).__name__,'message':str(e)[:150] if isinstance(e,(APIError,RuntimeError)) else 'redacted','at':time.time()}
        ledger.put('state',state)
        print(json.dumps(state['last_error']))
        return summary(state)
    finally:
        api.client.close()
        if (ledger.get('tick_lock') or {}).get('token')==token:
            ledger.pop('tick_lock',None)


@app.function(image=image,timeout=30)
def status():
    return summary(ledger.get('state') or fresh_state())
