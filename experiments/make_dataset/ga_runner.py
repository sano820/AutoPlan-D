# ga_runner.py
import os
import pickle
import time
import pandas as pd
import multiprocessing as mp
from tqdm.auto import tqdm 


import ga_util
from ga_core import GeneticAlgorithm

# --- "일꾼(Worker)" 함수 정의 ---
def ga_worker(args):
    """
    하나의 데이터 샘플에 대해 GA를 실행하고 결과를 반환하는 '일꾼' 함수입니다.
    verbose 플래그를 받아서 로그 출력 여부를 제어합니다.
    """
    # verbose 플래그를 인자에서 받음
    data_sample_df, fixed_hyperparams, sample_idx, verbose = args
    
    # verbose가 True일 때만 진행 상황 로그 출력
    if verbose:
        worker_pid = os.getpid()
        print(f"[Worker PID: {worker_pid}]가 샘플 {sample_idx + 1} 처리를 시작합니다.")
    
    try:
        # 데이터 전처리
        T_set, I_set, J_set, cit_data, pit_data, dit_data, mijt_data = \
            ga_util.preprocess_ga_inputs_dense(data_sample_df.copy())

        # GeneticAlgorithm 인스턴스 생성
        ga_instance = GeneticAlgorithm(
            T_set=T_set, I_set=I_set, J_set=J_set,
            cit_data=cit_data, pit_data=pit_data, dit_data=dit_data, mijt_data=mijt_data,
            n_pop=fixed_hyperparams['n_pop'], 
            n_iter=fixed_hyperparams['n_iter'],
            r_cross=fixed_hyperparams['r_cross'],
            r_mut=fixed_hyperparams['r_mut'],
            max_machine_work_time=fixed_hyperparams.get('max_machine_work_time', 600),
            overproduction_penalty_factor=fixed_hyperparams.get('overproduction_penalty_factor', 10000000),
            gene_swap_prob=fixed_hyperparams.get('gene_swap_prob', 0.5),
            xijt_keys_list=None 
        )
        
        # GA 실행
        best_solution_dict, best_score, log, log_detail = ga_instance.solve()
        
        # GT DataFrame 생성
        gt_df = pd.DataFrame(columns=['item', 'machine', 'time', 'qty'])
        if best_solution_dict:
            processed_gt_rows = []
            for (item_k, machine_k, time_k), ratio_val in best_solution_dict.items():
                demand = dit_data.get((item_k, time_k), 0) 
                actual_qty = round(ratio_val * demand)
                if actual_qty > 0:
                   processed_gt_rows.append({'item': item_k, 'machine': machine_k, 'time': time_k, 'qty': actual_qty})
            
            if processed_gt_rows:
                gt_df = pd.DataFrame(processed_gt_rows).sort_values(
                    by=['time', 'item', 'machine', 'qty']
                ).reset_index(drop=True)

        return (int(best_score), gt_df, log) # 반환할 결과들

    except Exception as e:
        worker_pid = os.getpid()
        print(f"[Worker PID: {worker_pid}]에서 오류! 샘플 {sample_idx + 1} 처리 중 문제 발생: {e}")
        return (None, pd.DataFrame(columns=['item', 'machine', 'time', 'qty']), [])

def generate_gt_in_parallel(dataset_for_ga_input, fixed_hyperparams, output_pickle_path=None, verbose=False):
    """
    multiprocessing.Pool을 사용하여 전체 데이터셋에 대해 GA를 병렬로 실행하고,
    결과 GT DataFrame 리스트를 지정된 경로의 pickle 파일로 저장합니다.
    verbose=True일 경우, 각 워커의 진행 상황 로그를 출력합니다.
    """
    num_processes = max(1, mp.cpu_count() - 2)
    print(f"병렬 처리에 사용할 CPU 코어(프로세스) 수: {num_processes}")
    
    tasks_to_run = [(data_sample, fixed_hyperparams, i, verbose) for i, data_sample in enumerate(dataset_for_ga_input)]
    
    print(f"\n총 {len(tasks_to_run)}개의 샘플에 대해 병렬 처리를 시작합니다...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(ga_worker, tasks_to_run), total=len(tasks_to_run)))
    
    # 결과 분리
    all_scores = [res[0] for res in results]
    all_gts = [res[1] for res in results]
    all_logs = [res[2] for res in results]
    
    print("모든 샘플에 대한 병렬 처리 완료!")

    # Pickle 저장 로직 
    if output_pickle_path: # 파일 경로가 주어졌을 경우에만 저장
        try:
            with open(output_pickle_path, 'wb') as f:
                pickle.dump(all_gts, f)
            print(f"\n최종 GT 리스트 저장 완료: {output_pickle_path}")
        except Exception as e:
            print(f"오류: Pickle 파일 저장 중 문제가 발생했습니다 - {e}")

    # 함수가 결과를 반환하는 좋은 패턴은 그대로 유지
    return all_scores, all_gts, all_logs