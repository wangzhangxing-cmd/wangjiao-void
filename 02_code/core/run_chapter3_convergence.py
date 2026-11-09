"""Reproducible C1-C3 studies; independent pressure histories for changed operators."""
from pathlib import Path
import csv, json, time, traceback, hashlib
import numpy as np
from coupled_void_nonassociated import CoupledVoidRockDomain
from local_void_interface import FreeSectorInterface
from nonaxisymmetric_nonassociated_domain import FullCircleRockDomain
from seal_stress_recovery import _surface_coefficients, von_mises
import run_b10_panel_convergence_9mpa as previous

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'chapter3_convergence_10MPa'
VERSION='chapter3-c1-c3-v1'
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)
def table(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(path,state):
    tmp=path.with_suffix('.tmp.npz');previous.save_state(tmp,state);tmp.replace(path)
def advance(domain,old,target,trace,depth=0):
    """Cut back only iteration failures; retain material parameters and tolerances."""
    try:
        state=domain.step(old,target,max_iterations=800,source_tolerance=1e-9,stress_tolerance_pa=500)
        trace.append(dict(pressure_mpa=target/1e6,iterations=int(state['iterations']),
                          source_gap=float(state['source_gap']),stress_mismatch_pa=float(state['stress_mismatch_pa'])))
        return state
    except RuntimeError as exc:
        if 'Coupled void iteration failed' not in str(exc) or depth>=4:raise
        midpoint=(old['internal_pressure_pa']+target)/2
        print(json.dumps(dict(status='cutback',from_mpa=old['internal_pressure_pa']/1e6,
                              to_mpa=target/1e6,midpoint_mpa=midpoint/1e6)),flush=True)
        middle=advance(domain,old,midpoint,trace,depth+1)
        return advance(domain,middle,target,trace,depth+1)
def sampling(interface,state,case):
    coeff=_surface_coefficients(interface,state['boundary']['lining_u'],1e7)
    rows=[]
    for count in (4096,8192,16384,32768,65536):
        maxima=[]
        for name,c in coeff.items():
            spectrum=np.zeros((count//2+1,4),complex)
            spectrum[:len(c)]=c*count/2;spectrum[0]=c[0]*count
            stress=np.fft.irfft(spectrum,n=count,axis=0)
            mises=von_mises(stress)/1e6;idx=int(np.argmax(mises))
            maxima.append((float(mises[idx]),name,idx*360/count))
            if count==65536:
                # Narrow physical endpoint profiles retain the actual full maximum.
                theta=np.arange(count)*360/count
                signed=(theta+180)%360-180
                mask=np.abs(np.abs(signed)-5)<3
                table(case/f'C3_{name}_endpoint_profile.csv',[
                    {'local_angle_deg':float(t),'mises_mpa':float(v)}
                    for t,v in zip(signed[mask],mises[mask])])
        peak=max(maxima)
        rows.append({'samples':count,'maximum_mises_mpa':peak[0],
                     'controlling_surface':peak[1],'local_angle_deg':peak[2],
                     'cavern_angle_deg':(peak[2]+90)%360})
    ref=rows[-1]['maximum_mises_mpa']
    for r in rows:r['relative_to_65536_percent']=100*(r['maximum_mises_mpa']/ref-1)
    table(case/'C3_sampling.csv',rows)
    return rows[-1]
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    source=ROOT/'results'/'void_10deg_panel_convergence_9MPa'
    kernel=dict(np.load(ROOT/'results'/'elastic_rock_kernel_nr64_nt512'/'kernel.npz'))
    rock=FullCircleRockDomain(kernel)
    with np.load(ROOT/'results'/'nonaxisymmetric_excavation_nr64_nt512'/'installation_state.npz') as f:
        installation={k:f[k].copy() for k in ('plastic','strain','stress','wall_displacement_m')}
    configs=[]
    for panels in (256,128,512,1024):
        configs.append(dict(id=f'C1_panels_{panels}',study='C1',panels=panels,core=1024,maximum=8192,wall=1024))
    for key,values in [('core',(512,2048)),('maximum',(4096,16384)),('wall',(512,2048))]:
        for v in values:
            c=dict(id=f'C2_{key}_{v}',study='C2',panels=256,core=1024,maximum=8192,wall=1024)
            c[key]=v;configs.append(c)
    # The original panel sequence changed the yield classification, so extend
    # it rather than claiming that 1024 panels is an asymptotic reference.
    configs.append(dict(id='C1_panels_2048',study='C1_extended',panels=2048,
                        core=1024,maximum=8192,wall=1024))
    for nr,nt in ((32,512),(32,256)):
        configs.append(dict(id=f'C4_nr{nr}_nt{nt}',study='C4',panels=256,
                            core=1024,maximum=8192,wall=1024,nr=nr,nt=nt))
    write(OUT/'study_plan.json',dict(version=VERSION,angle_deg=10,pressure_mpa=10,
        step_mpa=0.1,rock_mesh=[64,512],configs=configs,
        interpretation='10 MPa results are mathematical extension if seal exceeds 420 MPa; no applicable-domain validation error claim.',
        history='C1 continues its own verified 9 MPa history; changed C2 operators reload from installation.'))
    reports=[]
    previous.TARGET_PRESSURE_MPA=10.0
    for config in configs:
        case=OUT/config['id'];case.mkdir(exist_ok=True)
        if (case/'diagnostic_stop.json').exists():
            print(json.dumps(dict(case=config['id'],status='diagnostic_stop_retained')),flush=True)
            continue
        summary=case/'summary.json'
        if summary.exists():
            r=json.loads(summary.read_text(encoding='utf-8'))
            if r.get('complete') and r.get('version')==VERSION:reports.append(r);continue
        write(OUT/'progress.json',dict(case=config,stage='initializing',finished=len(reports),total=len(configs)))
        nr,nt=config.get('nr',64),config.get('nt',512)
        rock=FullCircleRockDomain(dict(np.load(ROOT/'results'/f'elastic_rock_kernel_nr{nr}_nt{nt}'/'kernel.npz')))
        with np.load(ROOT/'results'/f'nonaxisymmetric_excavation_nr{nr}_nt{nt}'/'installation_state.npz') as f:
            installation={k:f[k].copy() for k in ('plastic','strain','stress','wall_displacement_m')}
        interface=FreeSectorInterface(10,config['panels'],config['core'],config['maximum'])
        domain=CoupledVoidRockDomain(rock,installation,interface,config['wall'])
        progress=case/'progress.json';checkpoint=case/'latest_state.npz';records=[]
        if checkpoint.exists() and progress.exists():
            records=json.loads(progress.read_text(encoding='utf-8'))['records']
            state=previous.load_core_state(checkpoint)
            actual=state['internal_pressure_pa']/1e6
            if records and abs(records[-1]['pressure_mpa']-actual)>1e-8:raise RuntimeError('Checkpoint/progress pressure mismatch')
        elif config['study']=='C1':
            src=(ROOT/'results'/'void_10deg_integer_states'/'state_10MPa.npz') if config['panels']==256 else source/f"panels_{config['panels']}"/'state_09MPa.npz'
            state=previous.load_core_state(src)
            expected=10 if config['panels']==256 else 9
            if abs(state['internal_pressure_pa']/1e6-expected)>1e-8:raise RuntimeError('Source pressure mismatch')
            write(case/'source.json',dict(path=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
        else:state=domain.initial_state()
        began=time.perf_counter()
        for p in np.arange(0.1,10.0001,0.1):
            if p*1e6<=state['internal_pressure_pa']+0.01:continue
            trace=[]
            state=advance(domain,state,float(p*1e6),trace)
            record=dict(pressure_mpa=float(p),iterations=int(state['iterations']),source_gap=float(state['source_gap']),stress_mismatch_pa=float(state['stress_mismatch_pa']))
            record['accepted_substeps']=trace
            records.append(record);save(checkpoint,state)
            write(progress,dict(config=config,records=records))
            write(OUT/'progress.json',dict(case=config,stage='loading',pressure_mpa=float(p),finished=len(reports),total=len(configs)))
            print(json.dumps(dict(case=config['id'],**record)),flush=True)
        state['boundary']=domain.interface(1e7,state['plastic'])
        b=state['boundary']
        if np.min(b['panel_traction'][:,0])<0:raise RuntimeError('Tensile normal traction: review bonded-interface applicability')
        if b['linear_relative_residual']>1e-8:raise RuntimeError('Interface residual exceeds tolerance')
        previous.CORE_ORDER=config['core'];previous.MAXIMUM_ORDER=config['maximum'];previous.WALL_ORDER=config['wall']
        r=previous.final_metrics(domain,interface,state,installation,False,records,time.perf_counter()-began)
        r.update(config);r['version']=VERSION
        radii=rock.data['radii'];local=rock.local_components(state['plastic'])
        active=np.sqrt(np.sum(local[...,:3]**2,axis=-1)+2*local[...,3]**2)>1e-6
        occupied=np.any(active,axis=1)
        r['total_plastic_margin_m']=float(radii[-1]-np.max(radii[1:][occupied])) if np.any(occupied) else float(radii[-1]-radii[0])
        if np.any(active[-1]):raise RuntimeError('Plasticity reaches source-domain outer boundary')
        r['minimum_contact_pressure_mpa']=float(np.min(b['panel_traction'][:,0])/1e6)
        r['C3_high_resolution']=sampling(interface,state,case)
        r['seal_elastic_at_10MPa']=r['C3_high_resolution']['maximum_mises_mpa']<420
        save(case/'state_10MPa.npz',state);write(summary,r);reports.append(r)
        write(OUT/'partial_summary.json',dict(cases=reports))
        print(json.dumps(dict(case=config['id'],status='complete')),flush=True)
    stopped=[c['id'] for c in configs if (OUT/c['id']/'diagnostic_stop.json').exists()]
    write(OUT/'summary.json',dict(complete=len(reports)==len(configs),queue_finished=True,
                                diagnostic_stops=stopped,version=VERSION,cases=reports))
    write(OUT/'progress.json',dict(complete=len(reports)==len(configs),queue_finished=True,
                                 diagnostic_stops=stopped,finished=len(reports),total=len(configs)))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True,parents=True)
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
