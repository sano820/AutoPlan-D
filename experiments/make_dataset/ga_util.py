# ga_util.py

import pandas as pd 
import numpy as np  

def preprocess_ga_inputs_dense(dataset):
    """
    GA 입력 데이터를 전처리합니다 (Dense 방식).
    cit, pit, dit, mijt 딕셔너리는 I, T, J의 모든 조합에 대해 값을 가지며,
    원본 데이터에 해당 조합이 없으면 0으로 채웁니다.
    """
    I_unique = sorted(list(set(dataset['item'])))
    T_unique = sorted(list(set(dataset['time'])))
    J_unique = sorted(list(set(dataset['machine'])))

    idx_item_time = pd.MultiIndex.from_product([I_unique, T_unique], names=['item', 'time'])

    cit_series = dataset.groupby(['item', 'time'])['cost'].first()
    cit = cit_series.reindex(idx_item_time, fill_value=0).to_dict()

    pit_series = dataset.groupby(['item', 'time'])['urgent'].first()
    pit = pit_series.reindex(idx_item_time, fill_value=0).to_dict()

    # dit는 데이터 특성에 따라 .first() 또는 .sum() 사용
    dit_series = dataset.groupby(['item', 'time'])['qty'].first() 
    dit = dit_series.reindex(idx_item_time, fill_value=0).to_dict()

    # mijt 계산 (Dense 방식)
    idx_item_machine = pd.MultiIndex.from_product([I_unique, J_unique], names=['item', 'machine'])
    capacity_series_from_data = dataset.groupby(['item', 'machine'])['capacity'].first()
    # (item,machine) 조합에 대해 capacity가 0이거나 없으면 0으로 채움
    capacity_map_dense = capacity_series_from_data.reindex(idx_item_machine, fill_value=0) 
    
    mijt = {}
    for (i_item, j_machine), capacity_value in capacity_map_dense.items():
        # capacity_value가 0.0이면 정수 0으로, 아니면 원래 float 값 사용
        value_to_assign = 0 if capacity_value == 0.0 else capacity_value
        for t_time in T_unique:
            mijt[(i_item, j_machine, t_time)] = value_to_assign

    return T_unique, I_unique, J_unique, cit, pit, dit, mijt
    
def preprocess_ga_inputs_sparse(dataset):
    """
    GA 입력 데이터를 전처리합니다 (Sparse 방식).
    cit, pit, dit, mijt 딕셔너리는 원본 데이터에 실제로 존재하는 조합에 대해서만 값을 가집니다.
    GA 로직에서는 .get(key, 0)을 사용하여 없는 키에 접근해야 합니다.
    """
    I_unique = sorted(list(set(dataset['item'])))
    T_unique = sorted(list(set(dataset['time'])))
    J_unique = sorted(list(set(dataset['machine']))) # mijt 생성 시 필요할 수 있음

    # (item, time)으로 그룹핑하고 각 그룹의 첫 번째 값을 가져와 바로 딕셔너리로 변환
    cit = dataset.groupby(['item', 'time'])['cost'].first().to_dict()
    pit = dataset.groupby(['item', 'time'])['urgent'].first().to_dict()
    # dit는 데이터 특성에 따라 .first() 또는 .sum() 사용
    dit = dataset.groupby(['item', 'time'])['qty'].first().to_dict() 

    # mijt 계산 (Sparse 방식)
    # (item, machine) 조합의 capacity를 가져오되, 데이터에 있는 조합만 처리
    capacity_map_sparse = dataset.groupby(['item', 'machine'])['capacity'].first().to_dict()
    
    mijt = {}
    for (i_item, j_machine), capacity_value in capacity_map_sparse.items():
        # 여기서 capacity_value가 0인 것을 포함할지 여부는 네 결정에 따름.
        # 네가 "존재 하는 (아이템, 머신) 중에 실제로 capacity_value가 0인 것이 있을 수 있잖아. 그럼 이건 포함 해야 하지 않니"
        # 라고 했으니, capacity_value > 0 같은 필터 없이 그대로 사용.
        for t_time in T_unique: # T_unique는 전체 시간 범위를 사용
            mijt[(i_item, j_machine, t_time)] = capacity_value
            
    return T_unique, I_unique, J_unique, cit, pit, dit, mijt

def dict_to_list(data_dict, keys_list, default_value=0):
    """딕셔너리를 제공된 키 리스트의 순서에 따라 리스트로 변환 합니다.

    """
    return [data_dict.get(key, default_value) for key in keys_list]

def list_to_dict(data_list, keys_list):
    """리스트를 제공된 키 리스트에 매핑하여  딕셔너리로 변환 합니다.

    """
    if len(data_list) != len(keys_list):
        min_len = min(len(data_list), len(keys_list))
        print(f"Warning: list_to_dict에서 리스트와 키 리스트의 길이가 다릅니다. 짧은 쪽({min_len}개)에 맞춰 처리합니다.")
        return {keys_list[i]: data_list[i] for i in range(min_len)}

    return {key: data_list[i] for i, key in enumerate(keys_list)}

def set_hyper_parameters():
    # 하이퍼파라미터 설정
    hyper_parameters = pd.DataFrame({
        'index':    ['index_1', 'index_2', 'index_3', 'index_4',
                     'index_5', 'index_6', 'index_7', 'index_8'],
        'n_iter':   [500, 500, 500, 500, 500, 500, 500, 500],
        'n_pop':    [10, 20, 40, 20, 20, 20, 20, 20],
        'r_cross':  [0.4, 0.4, 0.4, 0.1, 0.2, 0.3, 0.4, 0.4],
        'r_mut':    [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.6]
    })

    # 결과 기록용 열 추가
    hyper_parameters['objective'] = np.nan # numpy 사용 확인 (import numpy as np 필요)
    hyper_parameters['time'] = np.nan    # numpy 사용 확인

    return hyper_parameters