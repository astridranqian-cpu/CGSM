import os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from .dataset import (
    load_merged_attributes_camels_us,
    assemble_snow_cluster_timeseries,
    slice_by_split,
    compute_norm_stats_source_train,
    apply_norm,
    fit_apply_static_scaler,
    select_static_feature_names,
    RegionalFusionWindowDataset,
    RegionalMixBatchSampler,
    collate_fn_verify,
)
from .modules.ric import compute_ric_weights
from .io import PROJECT,read_json

def basin_list(name):
    return (PROJECT/'splits'/name).read_text(encoding='utf-8').split()

def settings(config,data_root=None):
    c=dict(config['data']); c['root']=str(Path(data_root or os.environ.get('CAMELS_ROOT',c['root'])).resolve())
    if not Path(c['root']).is_dir(): raise FileNotFoundError(f"CAMELS directory not found: {c['root']}")
    return c

def loader_for(df,dyn_cols,stat_dict,cfg,train=False):
    ds=RegionalFusionWindowDataset(df,dyn_cols,stat_dict,cfg['look_back'])
    if not len(ds): raise ValueError('No valid input windows')
    kw=dict(collate_fn=collate_fn_verify,num_workers=cfg['num_workers'])
    if cfg['num_workers']>0: kw.update(persistent_workers=True,prefetch_factor=4)
    if train:
        sampler=RegionalMixBatchSampler(ds.get_indices_by_gauge(),cfg['batch_size'],cfg['seed'],True)
        return DataLoader(ds,batch_sampler=sampler,**kw)
    return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=False,drop_last=False,**kw)

def prepare_training(config,data_root=None):
    cfg=settings(config,data_root)
    source=basin_list('source_basins.txt'); target=basin_list('target_basins.txt')
    ids=sorted(source+target)
    if len(source)!=77 or len(target)!=20 or len(set(ids))!=97: raise ValueError('Expected 77 source and 20 target basins')
    schema=read_json(PROJECT/'configs/input_schema.json')
    static_cols=select_static_feature_names(schema['static_feature_names'])
    attrs=load_merged_attributes_camels_us(str(Path(cfg['root'])/cfg['attr_subdir']))
    missing=set(static_cols)-set(attrs.columns)
    if missing: raise ValueError(f'Missing static attributes: {sorted(missing)}')
    full,dyn_cols=assemble_snow_cluster_timeseries(ids,attrs,cfg)
    if set(full.gauge_id)!=set(ids): raise ValueError('Missing basin input records')
    if dyn_cols!=schema['dynamic_feature_names']: raise ValueError('Dynamic column order differs')
    mapping={}
    for gid in source:
        n=int((full.gauge_id==gid).sum())
        mapping[gid]=(int(n*cfg['temporal_train_ratio']),int(n*(cfg['temporal_train_ratio']+cfg['temporal_val_ratio'])),n)
    frames={m:slice_by_split(full,source,target,mapping,m) for m in ('train','val','local_test','target_all')}
    norms=compute_norm_stats_source_train(frames['train'],dyn_cols)
    static=attrs[attrs.gauge_id.isin(ids)][['gauge_id']+static_cols]
    stat_dict,scaler=fit_apply_static_scaler(static,static_cols,source,ids)
    ric=compute_ric_weights(frames['train'],dyn_cols,lag=config['model']['ric_lag_days'])
    prep={'source_basins':source,'target_basins':target,'static_columns':static_cols,
        'dyn_cols':dyn_cols,'norm_stats':norms,'static_mean':scaler.mean_.tolist(),
        'static_scale':scaler.scale_.tolist(),'ric':ric.tolist(),'data_cfg':cfg}
    loaders={m:loader_for(apply_norm(df,dyn_cols,norms),dyn_cols,stat_dict,cfg,m=='train') for m,df in frames.items()}
    return loaders,prep

def prepare_evaluation(config,prep,data_root=None,region='S0'):
    import pandas as pd
    cfg=settings(config,data_root)
    if region=='S0': ids=list(prep['target_basins'])
    elif region=='source_local_test': ids=list(prep['source_basins'])
    else:
        groups=pd.read_csv(PROJECT/'splits/regions_k5.csv',dtype={'gauge_id':str})
        ids=groups.loc[groups.region==region,'gauge_id'].tolist()
        if not ids: raise ValueError(f'Unknown region {region}')
    attrs=load_merged_attributes_camels_us(str(Path(cfg['root'])/cfg['attr_subdir']))
    frame,dyn_cols=assemble_snow_cluster_timeseries(ids,attrs,cfg)
    if set(frame.gauge_id)!=set(ids): raise ValueError('Missing evaluation basin records')
    if dyn_cols!=prep['dyn_cols']: raise ValueError('Dynamic column order differs')
    if region=='source_local_test':
        mapping={}
        for gid in ids:
            n=int((frame.gauge_id==gid).sum())
            mapping[gid]=(int(n*cfg['temporal_train_ratio']),int(n*(cfg['temporal_train_ratio']+cfg['temporal_val_ratio'])),n)
        frame=slice_by_split(frame,ids,[],mapping,'local_test')
    static_cols=select_static_feature_names(prep['static_columns'])
    if static_cols != list(prep['static_columns']):
        raise ValueError('Static feature order mismatch')
    if len(prep['static_mean'])!=40 or len(prep['static_scale'])!=40:
        raise ValueError('Expected 40 static normalization entries')
    sub=attrs[attrs.gauge_id.isin(ids)]
    values=np.nan_to_num(sub[static_cols].to_numpy(dtype=np.float64))
    values=((values-np.asarray(prep['static_mean']))/np.asarray(prep['static_scale'])).astype(np.float32)
    stat_dict={gid:values[i] for i,gid in enumerate(sub.gauge_id)}
    return loader_for(apply_norm(frame,dyn_cols,prep['norm_stats']),dyn_cols,stat_dict,cfg)
