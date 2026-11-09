"""Independent, checkpointed P0 jobs. Original programs and results are read-only.

Run without arguments for the queue, or --case CASE_ID for a single worker.
Each worker imports a frozen source snapshot, never another panel count's
operating plastic state. A single atomic checkpoint commits state, history,
and the pending adaptive substeps together.
"""
from pathlib import Path
import os,sys,json,csv,time,hashlib,shutil,traceback,argparse,subprocess
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'supplement'/'precision_P0_20261107'
SNAPSHOT=HERE/'core_snapshot'
VERSION='precision-p0-v1'
CASES=[('P0_3_b10_n4096',10,4096),('P0_1_b05_n1024',5,1024),
       ('P0_2_b15_n1024',15,1024),('P0_1_b05_n2048',5,2048),('P0_2_b15_n2048',15,2048)]

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def csvwrite(path,rows):
    if not rows:return
    tmp=Path(path).with_suffix('.tmp.csv')
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)
def event(**kw):print(json.dumps(dict(time=now(),**kw),ensure_ascii=False),flush=True)

def setup():
    OUT.mkdir(parents=True,exist_ok=True);SNAPSHOT.mkdir(parents=True,exist_ok=True)
    sources={}
    for p in sorted((ROOT/'02_code'/'core').glob('*.py')):
        digest=sha(p);dest=SNAPSHOT/p.name
        if dest.exists():
            if sha(dest)!=digest:raise RuntimeError('Frozen source differs from original: '+p.name)
        else:shutil.copy2(p,dest)
        sources[str(p.relative_to(ROOT))]=digest
    kernel=ROOT/'results'/'elastic_rock_kernel_nr64_nt512'/'kernel.npz'
    install=ROOT/'results'/'nonaxisymmetric_excavation_nr64_nt512'/'installation_state.npz'
    provenance=dict(created_utc=now(),original_source_sha256=sources,
                    kernel_path=str(kernel),kernel_sha256=sha(kernel),
                    installation_path=str(install),installation_sha256=sha(install),
                    existing_b10_n2048=str(ROOT/'results'/'chapter3_convergence_10MPa'/'C1_panels_2048'),
                    original_results_modified=False)
    provfile=OUT/'source_provenance.json'
    if provfile.exists():
        old=json.loads(provfile.read_text(encoding='utf-8'))
        for key in ['original_source_sha256','kernel_sha256','installation_sha256']:
            if old[key]!=provenance[key]:raise RuntimeError('Input provenance changed: '+key)
    else:write(provfile,provenance)
    for name,beta,panels in CASES:
        case=OUT/name;case.mkdir(exist_ok=True)
        config=dict(case_id=name,study='P0',void_angle_deg=beta,pressure_target_mpa=10,
            contact_panels=panels,core_order=1024,interface_order=8192,wall_order=1024,
            rock_nr=64,rock_nt=512,steel_stress_samples=65536,main_increment_mpa=.1,
            source_tolerance=1e-9,stress_tolerance_pa=500.,interface_relative_tolerance=1e-8,
            max_iterations=800,minimum_increment_mpa=.1/256,plastic_norm_threshold=1e-6,
            kernel_path=str(kernel),kernel_sha256=provenance['kernel_sha256'],
            installation_path=str(install),installation_sha256=provenance['installation_sha256'],
            geometry=dict(seal_inner_m=5.992,lining_inner_m=6.,lining_outer_m=6.5),
            steel=dict(E_pa=210e9,nu=.30,fy_pa=420e6,thickness_m=.008),
            lining=dict(E_pa=33.4e9,nu=.20,hoop_residual=.05,shear_residual=.05,
                        rebar_E_pa=200e9,inner_bar_diameter_m=.012,inner_spacing_m=.2,
                        outer_bar_diameter_m=.020,outer_spacing_m=.2,cover_m=.05),
            rock=dict(E_pa=12e9,nu=.26,cohesion_pa=.9e6,phi_deg=40.,psi_deg=10.,p0_pa=3.75e6),
            history='Matched 64x512 excavation installation state; each interface reloads independently from 0 MPa')
        path=case/'config.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=config:raise RuntimeError('Case config changed: '+name)
        else:write(path,config)
    return provenance

def worker(name):
    import numpy as np
    sys.path.insert(0,str(SNAPSHOT))
    from dataclasses import asdict
    from local_void_interface import FreeSectorInterface,arc_mean
    from coupled_void_nonassociated import CoupledVoidRockDomain
    from nonaxisymmetric_nonassociated_domain import FullCircleRockDomain
    from seal_stress_recovery import _surface_coefficients,von_mises
    case=OUT/name;configfile=case/'config.json';cfg=json.loads(configfile.read_text(encoding='utf-8'))
    signature=sha(configfile);records=[];pending=[];state=None
    write(case/'worker_identity.json',dict(pid=os.getpid(),started_utc=now(),runner_sha256=sha(__file__),config_sha256=signature))
    def status(stage,**kw):
        write(case/'progress.json',dict(case_id=name,stage=stage,pid=os.getpid(),updated_utc=now(),
            pressure_mpa=None if state is None else float(state['internal_pressure_pa'])/1e6,
            accepted_steps=len(records),**kw))
    try:
        if (case/'summary.json').exists():
            old=json.loads((case/'summary.json').read_text(encoding='utf-8'))
            if old.get('complete') and old['config_sha256']==signature:
                event(case=name,status='already_complete');return
        status('loading_matching_installation')
        assert sha(cfg['kernel_path'])==cfg['kernel_sha256']
        assert sha(cfg['installation_path'])==cfg['installation_sha256']
        with np.load(cfg['kernel_path']) as f:kernel={k:f[k] for k in f.files}
        rock=FullCircleRockDomain(kernel)
        assert (rock.nr,rock.nt)==(cfg['rock_nr'],cfg['rock_nt'])
        with np.load(cfg['installation_path']) as f:
            installation={k:f[k].copy() for k in ['plastic','strain','stress','wall_displacement_m']}
            assert float(f['wall_pressure_pa'])==0
        status('building_interface')
        event(case=name,status='building_interface',panels=cfg['contact_panels'])
        interface=FreeSectorInterface(cfg['void_angle_deg'],cfg['contact_panels'],cfg['core_order'],cfg['interface_order'])
        assert asdict(interface.op.layers.lining_input)==dict(inner_radius_m=6.,outer_radius_m=6.5,
            concrete_modulus_pa=33.4e9,concrete_poisson=.2,steel_modulus_pa=200e9,
            residual_hoop_ratio=.05,residual_shear_ratio=.05,cover_m=.05,
            inner_bar_diameter_m=.012,inner_bar_spacing_m=.2,outer_bar_diameter_m=.020,
            outer_bar_spacing_m=.2,include_reinforcement=True)
        write(case/'operator_parameters.json',dict(lining=asdict(interface.op.layers.lining_input),
            seal=asdict(interface.op.layers.seal_input),rock=cfg['rock'],
            core=interface.op.core_order,maximum=interface.order,panels=interface.panels,
            reciprocity_gap=interface.op.maximum_reciprocity_gap))
        domain=CoupledVoidRockDomain(rock,installation,interface,cfg['wall_order'])
        checkpoint=case/'checkpoint.npz'
        if checkpoint.exists():
            with np.load(checkpoint) as f:
                assert str(f['config_sha256'])==signature
                state={k:f[k].copy() for k in ['plastic','strain','stress']}
                state['internal_pressure_pa']=float(f['internal_pressure_pa'])
                records=json.loads(str(f['records_json']));pending=json.loads(str(f['pending_pa_json']))
            state['boundary']=domain.interface(state['internal_pressure_pa'],state['plastic'])
            event(case=name,status='resumed',pressure_mpa=state['internal_pressure_pa']/1e6)
        else:
            state=domain.initial_state();pending=[float(i*100000) for i in range(1,101)]
        def commit():
            temp=case/'checkpoint.tmp.npz';b=state['boundary']
            np.savez_compressed(temp,**{k:state[k] for k in ['plastic','strain','stress','internal_pressure_pa']},
                panel_traction=b['panel_traction'],traction_coefficients=b['traction_coefficients'],
                lining_u=b['lining_u'],rock_u=b['rock_u'],config_sha256=np.array(signature),
                records_json=np.array(json.dumps(records)),pending_pa_json=np.array(json.dumps(pending)))
            temp.replace(checkpoint)
            if records:
                csvwrite(case/'steps.csv',records)
                csvwrite(case/'plastic_area.csv',[{k:r[k] for k in ['pressure_mpa','operational_plastic_area_m2','total_plastic_margin_m']} for r in records])
        def quality(s):
            b=s['boundary'];q=b['panel_traction'];edges=interface.edges
            for k in ['plastic','strain','stress']:
                if not np.all(np.isfinite(s[k])):raise RuntimeError('Nonfinite state '+k)
            if not np.all(np.isfinite(q)):raise RuntimeError('Nonfinite interface traction')
            qmin=float(q[:,0].min());scale=max(float(abs(q).max()),1)
            fx=np.sum(q[:,0]*(np.sin(edges[1:])-np.sin(edges[:-1]))-q[:,1]*(np.cos(edges[:-1])-np.cos(edges[1:])))
            fy=np.sum(q[:,0]*(np.cos(edges[:-1])-np.cos(edges[1:]))+q[:,1]*(np.sin(edges[1:])-np.sin(edges[:-1])))
            torque=np.sum(q[:,1]*np.diff(edges));balance=float(max(abs(fx),abs(fy),abs(torque))/scale)
            total=s['plastic'];norm=np.sqrt(np.sum(total[:,:3]**2,axis=1)+2*total[:,3]**2).reshape(rock.nr,rock.nt)
            occupied=np.any(norm>cfg['plastic_norm_threshold'],axis=1)
            margin=float(kernel['radii'][-1]-max(kernel['radii'][1:][occupied])) if occupied.any() else float(kernel['radii'][-1]-kernel['radii'][0])
            delta=total-installation['plastic'];dnorm=np.sqrt(np.sum(delta[:,:3]**2,axis=1)+2*delta[:,3]**2)
            areas=np.repeat(np.pi*(kernel['radii'][1:]**2-kernel['radii'][:-1]**2)/rock.nt,rock.nt)
            physical=(s['strain']-s['plastic'])@rock.C.T-np.tile([rock.p0,rock.p0,rock.p0,0.],(rock.n,1))
            mismatch=float(abs(physical-s['stress']).max())
            if qmin < -1e-5:raise RuntimeError('Tensile normal traction; review bonded interface applicability')
            if balance>cfg['interface_relative_tolerance'] or b['linear_relative_residual']>cfg['interface_relative_tolerance']:raise RuntimeError('Interface balance/residual limit exceeded')
            if margin<=0 or occupied[-1]:raise RuntimeError('Plastic strain reaches source-domain outer boundary')
            if mismatch>cfg['stress_tolerance_pa']:raise RuntimeError('Independent stress recovery mismatch exceeds tolerance')
            return dict(minimum_contact_pressure_mpa=qmin/1e6,mean_contact_pressure_mpa=float(q[:,0].mean()/1e6),
                mean_void_displacement_mm=float(arc_mean(b['lining_u'],-interface.alpha,interface.alpha)[0]*1000),
                mean_contact_displacement_mm=float(arc_mean(b['lining_u'],interface.alpha,2*np.pi-interface.alpha)[0]*1000),
                force_torque_relative_residual=balance,interface_relative_residual=float(b['linear_relative_residual']),
                weak_displacement_gap_m=float(b['weak_displacement_gap_m']),
                independent_stress_mismatch_pa=mismatch,total_plastic_margin_m=margin,
                operational_plastic_area_m2=float(areas[dnorm>cfg['plastic_norm_threshold']].sum()))
        initial_quality=quality(state);commit();status('loading',next_pressure_mpa=pending[0]/1e6 if pending else None)
        began=time.perf_counter()
        while pending:
            target=pending[0];old_pressure=state['internal_pressure_pa'];increment=target-old_pressure
            status('solving_increment',next_pressure_mpa=target/1e6,increment_mpa=increment/1e6)
            try:
                candidate=domain.step(state,target,max_iterations=cfg['max_iterations'],
                    source_tolerance=cfg['source_tolerance'],stress_tolerance_pa=cfg['stress_tolerance_pa'])
            except RuntimeError as exc:
                recoverable=('Coupled void iteration failed' in str(exc) or 'Coupled constitutive mismatch' in str(exc))
                if not recoverable or increment/2<cfg['minimum_increment_mpa']*1e6-1e-8:raise
                midpoint=(old_pressure+target)/2;pending[:1]=[midpoint,target]
                with (case/'cutbacks.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=now(),from_pa=old_pressure,target_pa=target,retry_pa=midpoint,reason=str(exc)))+'\n')
                event(case=name,status='cutback',from_mpa=old_pressure/1e6,target_mpa=target/1e6,retry_mpa=midpoint/1e6)
                commit();continue
            numeric=quality(candidate)
            state=candidate;pending.pop(0)
            record=dict(pressure_mpa=target/1e6,increment_mpa=increment/1e6,iterations=int(state['iterations']),
                source_gap=float(state['source_gap']),stress_mismatch_pa=float(state['stress_mismatch_pa']),
                yield_residual_pa=float(state['yield_residual_pa']),stationarity_relative=float(state['stationarity_relative']),
                accepted_utc=now(),**numeric)
            records.append(record);commit();status('loading',last_step=record)
            event(case=name,status='accepted',pressure_mpa=record['pressure_mpa'],iterations=record['iterations'],elapsed_seconds=time.perf_counter()-began)
        status('recovering_final_fields')
        count=cfg['steel_stress_samples'];theta=np.arange(count)*2*np.pi/count
        local=(np.rad2deg(theta)+180)%360-180;global_angle=(np.rad2deg(theta)+90)%360
        coefficients=_surface_coefficients(interface,state['boundary']['lining_u'],state['internal_pressure_pa'])
        stresses={};surfaces={}
        for label,c in coefficients.items():
            spectrum=np.zeros((count//2+1,4),complex);spectrum[:len(c)]=c*count/2;spectrum[0]=c[0]*count
            values=von_mises(np.fft.irfft(spectrum,n=count,axis=0))/1e6;stresses[label]=values
            i=int(np.argmax(values));surfaces[label]=dict(maximum_mises_mpa=float(values[i]),local_angle_deg=float(local[i]),
                global_angle_deg=float(global_angle[i]),endpoint_offset_deg=float(abs(local[i])-cfg['void_angle_deg']/2))
        envelope=np.maximum(stresses['inner'],stresses['outer']);idx=int(np.argmax(envelope))
        controlling='inner' if stresses['inner'][idx]>=stresses['outer'][idx] else 'outer'
        peak=dict(maximum_mises_mpa=float(envelope[idx]),controlling_surface=controlling,
            local_angle_deg=float(local[idx]),global_angle_deg=float(global_angle[idx]),
            endpoint_offset_deg=float(abs(local[idx])-cfg['void_angle_deg']/2),samples=count)
        order=np.argsort(global_angle)
        csvwrite(case/'angular_stress.csv',[dict(theta_deg=float(global_angle[i]),local_angle_deg=float(local[i]),
            inner_mises_mpa=float(stresses['inner'][i]),outer_mises_mpa=float(stresses['outer'][i]),
            maximum_mises_mpa=float(envelope[i])) for i in order])
        width=min(3,cfg['void_angle_deg']*.4);endpoint=[]
        for side in [-1,1]:
            centre=side*cfg['void_angle_deg']/2;ids=np.flatnonzero(abs(local-centre)<=width)
            j=int(ids[np.argmax(envelope[ids])]);endpoint.append(dict(endpoint_local_deg=centre,
                maximum_mises_mpa=float(envelope[j]),peak_local_deg=float(local[j]),
                peak_global_deg=float(global_angle[j]),endpoint_offset_deg=float(side*local[j]-cfg['void_angle_deg']/2)))
        mids=(np.rad2deg(interface.mid)+90)%360;q=state['boundary']['panel_traction']
        csvwrite(case/'angular_contact.csv',[dict(theta_deg=float(mids[i]),local_midpoint_deg=float(np.rad2deg(interface.mid[i])),
            local_start_deg=float(np.rad2deg(interface.edges[i])),local_end_deg=float(np.rad2deg(interface.edges[i+1])),
            normal_pressure_mpa=float(q[i,0]/1e6),shear_traction_mpa=float(q[i,1]/1e6)) for i in np.argsort(mids)])
        delta=state['plastic']-installation['plastic'];dnorm=np.sqrt(np.sum(delta[:,:3]**2,axis=1)+2*delta[:,3]**2)
        areas=np.repeat(np.pi*(kernel['radii'][1:]**2-kernel['radii'][:-1]**2)/rock.nt,rock.nt);active=np.flatnonzero(dnorm>1e-6)
        points=kernel['points']
        csvwrite(case/'plastic_cells.csv',[dict(x_m=float(-points[i,1]),y_m=float(points[i,0]),cell_area_m2=float(areas[i]),plastic_increment_norm=float(dnorm[i])) for i in active])
        final=quality(state)
        summary=dict(case_id=name,complete=True,converged=True,version=VERSION,config_sha256=signature,
            runner_sha256=sha(__file__),beta_deg=cfg['void_angle_deg'],pressure_mpa=state['internal_pressure_pa']/1e6,
            contact_panels=cfg['contact_panels'],rock_nr=rock.nr,rock_nt=rock.nt,core_order=cfg['core_order'],
            interface_order=cfg['interface_order'],wall_order=cfg['wall_order'],**final,
            steel_peak=peak,steel_surface_peaks=surfaces,endpoint_peaks=endpoint,
            linear_steel_mathematical_extension=bool(peak['maximum_mises_mpa']>420),
            accepted_steps=len(records),adaptive_substeps=any(abs(r['increment_mpa']-.1)>1e-8 for r in records),
            maximum_source_gap=max(r['source_gap'] for r in records),
            maximum_stress_mismatch_pa=max(r['stress_mismatch_pa'] for r in records),
            created_utc=now(),reference_to_exact_solution=False)
        csvwrite(case/'summary.csv',[dict(case_id=name,beta_deg=summary['beta_deg'],contact_panels=summary['contact_panels'],
            pressure_mpa=summary['pressure_mpa'],**final,steel_peak_mpa=peak['maximum_mises_mpa'],
            steel_controlling_surface=controlling,steel_peak_global_angle_deg=peak['global_angle_deg'],steel_samples=count,
            mathematical_extension=summary['linear_steel_mathematical_extension'],converged=True)])
        # Keep the complete, atomic state as the result, with its full actual pressure history.
        shutil.copy2(checkpoint,case/'state_10MPa.npz')
        write(case/'summary.json',summary);status('complete',summary=summary)
        event(case=name,status='complete',summary=summary)
    except Exception as exc:
        failure=dict(case_id=name,converged=False,error=repr(exc),traceback=traceback.format_exc(),
            actual_last_pressure_mpa=None if state is None else state['internal_pressure_pa']/1e6,
            pending_pressure_mpa=pending[0]/1e6 if pending else None,accepted_steps=len(records),time=now())
        write(case/'failure.json',failure);status('failed',failure=failure);event(case=name,status='failed',error=repr(exc))
        raise


def queue(workers):
    setup();OUT.mkdir(exist_ok=True)
    pidfile=OUT/'run_pid.txt'
    if pidfile.exists():
        import ctypes
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
        api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        oldpid=int(pidfile.read_text())
        if oldpid!=os.getpid():
            handle=api.OpenProcess(0x1000,False,oldpid)
            if handle:
                code=wintypes.DWORD();ok=api.GetExitCodeProcess(handle,ctypes.byref(code));api.CloseHandle(handle)
                if ok and code.value==259:raise RuntimeError('Prior queue PID is active; inspect before starting another queue')
    pidfile.write_text(str(os.getpid()))
    todo=[];finished=[];failed=[];running={}
    for name,_,_ in CASES:
        if (OUT/name/'summary.json').exists():finished.append(name)
        elif (OUT/name/'failure.json').exists():failed.append(name)
        else:todo.append(name)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    while todo or running:
        while todo and len(running)<workers:
            name=todo.pop(0);log=(OUT/name/'run.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(__file__),'--case',name],cwd=str(HERE),env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            running[name]=(p,log);event(status='started',case=name,pid=p.pid)
        for name,(p,log) in list(running.items()):
            rc=p.poll()
            if rc is not None:
                log.close();del running[name]
                if rc==0 and (OUT/name/'summary.json').exists():finished.append(name)
                else:
                    failed.append(name)
                    if not (OUT/name/'failure.json').exists():write(OUT/name/'failure.json',dict(error='Worker exited',exit_code=rc,time=now()))
                event(status='worker_exited',case=name,exit_code=rc)
        write(OUT/'run_manifest.json',dict(version=VERSION,driver_pid=os.getpid(),updated_utc=now(),workers=workers,
            running={name:p.pid for name,(p,_) in running.items()},queued=todo,finished=finished,failed=failed,
            complete=len(finished)==5,queue_finished=not(todo or running),p1_status='Not started; evaluate P0 results first'))
        if todo or running:time.sleep(10)
    # Scientific source files and read-only input files must still match their initial hashes.
    prov=json.loads((OUT/'source_provenance.json').read_text(encoding='utf-8'))
    unchanged=all(sha(ROOT/name)==digest for name,digest in prov['original_source_sha256'].items())
    unchanged=unchanged and sha(prov['kernel_path'])==prov['kernel_sha256'] and sha(prov['installation_path'])==prov['installation_sha256']
    write(OUT/'queue_summary.json',dict(complete=len(finished)==5,finished=finished,failed=failed,
        original_sources_and_inputs_unchanged=unchanged,time=now(),p1_status='Pending result-based selection'))
    event(status='queue_finished',finished=finished,failed=failed,original_inputs_unchanged=unchanged)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.case:worker(args.case)
    else:queue(args.workers)
