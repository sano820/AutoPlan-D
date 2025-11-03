import random
import numpy as np 
import ga_util 

class GeneticAlgorithm:
    def __init__(self, T_set, I_set, J_set, cit_data, pit_data, dit_data, mijt_data,
                 n_pop, n_iter, r_cross, r_mut,
                 xijt_keys_list=None,
                 max_machine_work_time=600,
                 overproduction_penalty_factor=10000000,
                 gene_swap_prob=0.5):
        """
        유전자 알고리즘 초기화.

        Args:
            T_set (list): 생산 기한 날짜 집합. (정렬된 리스트 권장)
            I_set (list): 품목 종류 집합. (정렬된 리스트 권장)
            J_set (list): 기계 종류 집합. (정렬된 리스트 권장)
            cit_data (dict): (item, time)을 키로 하는 미생산 시 비용 딕셔너리.
            pit_data (dict): (item, time)을 키로 하는 긴급생산 필요 여부 딕셔너리.
            dit_data (dict): (item, time)을 키로 하는 요구량 딕셔너리.
            mijt_data (dict): (item, machine, time)을 키로 하는 생산 능력 딕셔너리.
            n_pop (int): 세대당 염색체(해) 수.
            n_iter (int): 반복할 세대 수.
            r_cross (float): 교배율.
            r_mut (float): 변이율.
            xijt_keys_list (list, optional): 염색체(xijt) 딕셔너리의 키 순서를 정의하는 (i,j,t) 튜플 리스트.
                                            None이면 I_set, J_set, T_set의 모든 조합으로 자동 생성됩니다.
            max_machine_work_time (int, optional): 기계의 일일 최대 작동 시간(분). 기본값은 600.
            overproduction_penalty_factor (float or int, optional): 과잉생산 시 단위당 페널티 계수. 기본값은 10,000,000.
            gene_swap_prob (float, optional): 교차가 발생할 경우, 각 유효 유전자 위치에서 실제로 유전자를 교환할 확률. 기본값은 0.5.
        """
        # 데이터 저장
        self.I_set = I_set
        self.J_set = J_set
        self.T_set = T_set
        self.cit = cit_data
        self.pit = pit_data
        self.dit = dit_data
        self.mijt = mijt_data

        # 하이퍼파라미터 저장
        self.n_pop = int(n_pop)
        self.n_iter = int(n_iter)
        self.r_cross = float(r_cross)
        self.r_mut = float(r_mut)
        self.MAX_MACHINE_WORK_TIME = int(max_machine_work_time)
        self.OVERPRODUCTION_PENALTY_FACTOR = float(overproduction_penalty_factor)
        self.GENE_SWAP_PROB = float(gene_swap_prob)
        
        # xijt_keys 초기화 (모든 가능한 (i,j,t) 조합)
        if xijt_keys_list:
            self.xijt_keys = xijt_keys_list
        else:
            if not (self.I_set and self.J_set and self.T_set): # 세트 중 하나라도 비어있으면
                print("Warning: I_set, J_set, 또는 T_set 중 하나 이상이 비어있어 xijt_keys를 생성할 수 없습니다. 빈 리스트로 설정됩니다.")
                self.xijt_keys = []
            else:
                # I_set, J_set, T_set이 정렬되어 있다면 생성되는 키 순서도 일정해짐.
                self.xijt_keys = [(i, j, t) for i in self.I_set for j in self.J_set for t in self.T_set]
        
        # mijt_keys는 xijt_keys와 동일한 구조 및 순서를 사용한다고 가정 (사용자 확인 사항).
        # 이는 mijt 딕셔너리 (self.mijt)에 접근할 때도 xijt_keys와 동일한 키 순서를 참조할 수 있게 함.
        # 만약 mijt_keys가 self.mijt 딕셔너리의 실제 키만을 나타내야 한다면 (즉, sparse 할 수 있다면)
        # self.mijt_keys = list(self.mijt.keys()) 로 설정할 수도 있지만,
        # xijt_keys와 동일하게 모든 (i,j,t) 조합을 가지는 것이 일관성 있을 수 있음.
        # 여기서는 xijt_keys와 동일하게 사용.
        self.mijt_keys = self.xijt_keys # mijt_keys도 xijt_keys와 동일한 전체 (i,j,t) 조합 사용


        # 나머지 인스턴스 변수 초기화
        self.population = [] # List of xijt dictionaries
        self.best_solution = None # An xijt dictionary
        self.best_score = float('inf') 
        self.log = [] # List of (generation, best_score)
        self.log_detail = [] # List of (generation, best_solution_dict, best_score)

        
    def _generate_initial_solution(self): 
        """
        하나의 초기 해(염색체 xijt 딕셔너리)를 생성합니다.
        각 (품목, 시간) 조합에 대해, 관련된 모든 기계들의 생산 비율의 합이
        1.0을 넘지 않도록 정규화된 랜덤 비율을 할당합니다.
        """
        xijt = {}
        
        # 1. 모든 키를 (품목, 시간) 쌍으로 먼저 그룹핑합니다.
        keys_by_item_time = {}
        for key_tuple in self.xijt_keys:
            # key_tuple = (item, machine, time)
            item_key, machine_key, time_key = key_tuple
            it_pair = (item_key, time_key)
            if it_pair not in keys_by_item_time:
                keys_by_item_time[it_pair] = []
            keys_by_item_time[it_pair].append(key_tuple)

        # 2. 각 (품목, 시간) 그룹에 대해 정규화된 비율을 할당합니다.
        for it_pair, key_list_for_machines in keys_by_item_time.items():
            
            # 해당 (품목, 시간)에 대한 요구량이 없으면 모든 기계의 생산 비율을 0으로 설정
            if self.dit.get(it_pair, 0) <= 0:
                for key_tuple in key_list_for_machines:
                    xijt[key_tuple] = 0.0
                continue
            
            # 요구량이 있다면, 각 기계에 대해 랜덤 숫자를 생성
            random_values = [random.random() for _ in key_list_for_machines]
            total_sum = sum(random_values)
            
            if total_sum > 0:
                # 모든 랜덤 숫자의 합으로 각 숫자를 나눠서, 전체 합이 1.0이 되도록 정규화
                # 이렇게 하면 모든 기계의 생산 비율 합이 정확히 100%가 됨
                # (약간의 미생산을 허용하고 싶다면, target_sum 같은 변수를 추가로 곱해줄 수도 있음)
                normalized_ratios = [val / total_sum for val in random_values]
                
                for i, key_tuple in enumerate(key_list_for_machines):
                    xijt[key_tuple] = normalized_ratios[i]
            else:
                # 모든 랜덤 숫자가 0인 드문 경우, 모든 비율을 0으로 설정
                for key_tuple in key_list_for_machines:
                    xijt[key_tuple] = 0.0
        
        # 3. 마지막으로 decode 메소드를 통해 한번 정제합니다.
        return self._decode(xijt)

    def _decode(self, individual_xijt): 
        """임의의 해인 individual_xijt 조합 중 범위 밖의 값을 0으로 조정하는 함수.
        mijt가 0인 것은 시간 t에, item i를 NC machine j로 만들 수 없다는 의미이므로 xijt 값을 0으로 할당.
        ---
        염색체(individual_xijt)를 기계 생산 능력(self.mijt) 제약에 맞춰 조정합니다.
        생산 능력이 0인 (품목, 기계, 시간) 조합의 생산 비율을 0으로 설정합니다.
        """
        decoded_xijt = {}
        
        # __init__에서 self.xijt_keys가 모든 (i,j,t) 조합으로 확실히 초기화되었다고 가정합니다.
        # self.mijt는 (i,j,t)를 키로 하는 생산 능력 딕셔너리입니다.
        # (preprocess_ga_inputs_dense를 사용했다면 모든 키에 대한 값을,
        #  sparse를 사용했다면 있는 키에 대한 값만 가질 수 있지만, .get으로 안전하게 접근합니다.)

        for key_tuple in self.xijt_keys: # 모든 정의된 유전자 위치를 순회
            # key_tuple은 (item, machine, time) 형태
            
            # 해당 위치의 생산 능력을 self.mijt에서 가져옴 (없으면 0)
            capacity = self.mijt.get(key_tuple, 0)
            
            if capacity == 0:
                decoded_xijt[key_tuple] = 0
            else:
                # 생산 능력이 있다면, 원래 individual_xijt의 값을 s사용 (없으면 0)
                decoded_xijt[key_tuple] = individual_xijt.get(key_tuple, 0)
                
        return decoded_xijt

    def _check_constraints(self, xijt): 
        """기계의 일일 최대 작동 시간(예:600분)과 같은 제약 조건을 만족하는지 확인하고,
        위반 시 해를 조정하는 함수.
        """
        checked_xijt = xijt.copy() # 원본 해를 변경하지 않기 위해 복사본 사용.

        for j_key in self.J_set: # self.J_set 직접 사용
            for t_key in self.T_set: # self.T_set 직접 사용
                total_production_time_for_machine_at_time = 0
                for i_key in self.I_set: # self.I_set 직접 사용
                    qty_produced = round(checked_xijt.get((i_key, j_key, t_key), 0) * self.dit.get((i_key, t_key), 0))
                    capacity = self.mijt.get((i_key, j_key, t_key), 0)
                    if capacity > 0 and qty_produced > 0:
                        # _check_constraints 메소드에서 수정되어야 할 부분 (만약 self.mijt가 "Time per Unit"이라면)
                        total_production_time_for_machine_at_time += qty_produced * self.mijt.get((i_key, j_key, t_key), 0)
                
                # 하드코딩된 600 대신 self.MAX_MACHINE_WORK_TIME 사용
                if total_production_time_for_machine_at_time > self.MAX_MACHINE_WORK_TIME: 
                    for i_key in self.I_set: # self.I_set 직접 사용
                        checked_xijt[(i_key, j_key, t_key)] = 0
        return checked_xijt

    def _calculate_objective_value(self, decoded_xijt): 
        """
        주어진 해(생산 계획)의 적합도(목적 함수 값)를 계산합니다.
        목표: (긴급 품목의 미생산 페널티) + (과잉생산 페널티)의 합을 최소화.
        총생산량은 모든 기계의 생산량을 합한 후 round()를 사용하여 계산합니다.
        """
        # 1. 제약조건 검사
        constrained_xijt = self._check_constraints(decoded_xijt)
        
        uit = {} 

        # 2. 각 (품목, 시간) 조합에 대해 'u' (요구량 - 생산량) 계산
        for i_key in self.I_set:
            for t_key in self.T_set:
                demand_qty = self.dit.get((i_key, t_key), 0)

                # 모든 기계의 생산량을 실수 형태로 먼저 모두 더함
                produced_qty_sum_float = 0.0
                for j_key in self.J_set:
                    production_ratio = constrained_xijt.get((i_key, j_key, t_key), 0)
                    produced_qty_for_machine = production_ratio * demand_qty 
                    produced_qty_sum_float += produced_qty_for_machine

                # 총합을 마지막에 한 번만 반올림
                produced_qty_sum_rounded = round(produced_qty_sum_float)
                
                # 순수 미생산량 (음수면 과잉생산)
                u = demand_qty - produced_qty_sum_rounded 
                
                # 3. uit 딕셔너리 채우기 (오리지널 로직)
                if u >= 0: # 미생산 또는 정확히 생산
                    uit[(i_key, t_key)] = u
                else: # 과잉 생산 (u가 음수)
                    # self.OVERPRODUCTION_PENALTY_FACTOR는 __init__에서 10000000으로 설정되어 있다고 가정
                    uit[(i_key, t_key)] = abs(u) * self.OVERPRODUCTION_PENALTY_FACTOR 
        
        # 4. 최종 목적 함수 값 계산 (오리지널 로직)
        objective_value = 0
        for i_key in self.I_set:
            for t_key in self.T_set:
                # uit.get()으로 안전하게 접근, 만약 uit에 해당 키가 없으면 0으로 처리 (이론상 모든 i,t 키가 uit에 있어야 함)
                u_it_value = uit.get((i_key, t_key), 0) 
                cost_it = self.cit.get((i_key, t_key), 0)
                priority_it = self.pit.get((i_key, t_key), 0)
                
                objective_value += u_it_value * cost_it * priority_it
                
        return objective_value

    def _perform_selection(self, scores):
        """
        현재 세대의 모집단(self.population)과 각 해의 점수(scores)를 바탕으로,
        오리지널 selection 함수의 로직을 사용하여 다음 세대를 구성할 
        self.n_pop개의 후보 해를 선택합니다.

        Args:
            scores (list): 현재 모집단(self.population)의 각 해에 대한 목적 함수 값(점수) 리스트.

        Returns:
            list: 선택된 해(염색체 딕셔너리)들로 구성된 새로운 모집단 리스트.
                  각 해는 원본의 깊은 복사본입니다.
        """
        new_selected_population = []
        current_pop_size = len(self.population)

        if current_pop_size == 0:
            # TODO: 현재 모집단이 비어있는 극단적인 경우에 대한 처리 (예: 경고 후 빈 리스트 반환)
            print("Warning: 현재 모집단이 비어있어 선택을 진행할 수 없습니다.")
            return []

        # k: 토너먼트 참가자 수 (오리지널 로직: self.n_pop의 절반)
        # self.n_pop은 다음 세대 모집단 크기 (클래스의 하이퍼파라미터)
        k_tournament_participants = round(self.n_pop * 0.5)
        # k_tournament_participants는 최소 1 이상이어야 의미가 있음 (1이면 그냥 랜덤 선택)
        if k_tournament_participants < 1:
            k_tournament_participants = 1
        
        for _ in range(self.n_pop): # 다음 세대 모집단 크기(self.n_pop)만큼 해를 선택
            if current_pop_size == 0: break # 방어 코드

            # --- 오리지널 selection 함수의 로직 시작 (하나의 해 선택) ---
            
            # 1. 초기 후보 인덱스 랜덤 선택
            #    numpy.random.randint(N)은 [0, N-1] 범위에서 정수 하나 반환
            #    만약 current_pop_size가 0이면 에러날 수 있으므로 위에서 방어.
            selected_candidate_idx = np.random.randint(current_pop_size)
            
            # 2. (k - 1) 명의 추가 경쟁자들과 비교
            num_additional_competitors = k_tournament_participants - 1
            
            if num_additional_competitors > 0:
                # 오리지널 코드: for ix in randint(0, len(pop), k - 1):
                # 이는 k-1개의 랜덤 인덱스를 생성하여 순회하는 것을 의미.
                # numpy.random.randint(low, high, size)는 [low, high-1) 범위에서 size개 정수 반환.
                # (만약 current_pop_size가 0이면 np.random.randint(0,0,...) 에러 발생 가능)
                if current_pop_size > 0:
                    # num_additional_competitors가 current_pop_size보다 훨씬 클 수도 있지만,
                    # np.random.randint는 중복을 허용하여 인덱스를 뽑음.
                    additional_competitor_indices = np.random.randint(0, current_pop_size, 
                                                                      size=num_additional_competitors)
                    for competitor_idx in additional_competitor_indices:
                        if scores[competitor_idx] < scores[selected_candidate_idx]:
                            selected_candidate_idx = competitor_idx
            
            # --- 오리지널 selection 함수의 로직 끝 ---
            
            # 선택된 가장 좋은 해를 새 모집단에 추가 (깊은 복사)
            new_selected_population.append(self.population[selected_candidate_idx].copy())
            
        return new_selected_population

    def _perform_crossover(self, parent1_dict, parent2_dict):
        """
        두 부모 해에 대해 산술 교차(Arithmetic Crossover)와 유사한 방식을 적용합니다.
        각 (품목, 시간) 그룹별로 비율을 교차하여, 자식의 비율 합도 1.0을 유지하도록 합니다.
        """
        child1_dict = {}
        child2_dict = {}
        
        # self.xijt_keys를 (item, time)으로 그룹핑 (효율을 위해 __init__에서 미리 해둘 수도 있음)
        keys_by_item_time = {}
        for key_tuple in self.xijt_keys:
            it_pair = (key_tuple[0], key_tuple[2]) # (item, time)
            if it_pair not in keys_by_item_time:
                keys_by_item_time[it_pair] = []
            keys_by_item_time[it_pair].append(key_tuple)

        if random.random() < self.r_cross: # 전체 교차 발생 확률
            for it_pair, key_list_for_machines in keys_by_item_time.items():
                # 교배 가중치 (alpha) 랜덤 선택
                alpha = random.random()

                for key_tuple in key_list_for_machines:
                    p1_val = parent1_dict.get(key_tuple, 0)
                    p2_val = parent2_dict.get(key_tuple, 0)
                    
                    # 산술 교차 적용
                    child1_dict[key_tuple] = alpha * p1_val + (1.0 - alpha) * p2_val
                    child2_dict[key_tuple] = alpha * p2_val + (1.0 - alpha) * p1_val
        else:
            # 교차가 일어나지 않으면 부모를 그대로 자식으로
            child1_dict = parent1_dict.copy()
            child2_dict = parent2_dict.copy()

        return child1_dict, child2_dict
        
    def _apply_mutation(self, chromosome_dict):
        """
        염색체에 변이를 적용합니다. self.r_mut 확률로 변이가 발생하며,
        (품목, 시간) 그룹 전체의 생산 비율을 재분배하여 비율의 합이 1.0인 제약조건을 유지합니다.
        """
        mutated_chromosome = chromosome_dict.copy() # 원본 수정을 피하기 위해 복사본에서 작업

        # self.xijt_keys를 (item, time)으로 그룹핑
        keys_by_item_time = {}
        for key_tuple in self.xijt_keys:
            it_pair = (key_tuple[0], key_tuple[2])
            if it_pair not in keys_by_item_time:
                keys_by_item_time[it_pair] = []
            keys_by_item_time[it_pair].append(key_tuple)
        
        # 각 (item, time) 그룹에 대해 r_mut 확률로 변이 시도
        for it_pair, key_list_for_machines in keys_by_item_time.items():
            if random.random() < self.r_mut:
                # 변이가 결정되면, 해당 (item, time)의 생산 비율을 완전히 새로 생성 (정규화 포함)
                if self.dit.get(it_pair, 0) > 0:
                    # _generate_initial_solution의 일부 로직 재사용
                    random_values = [random.random() for _ in key_list_for_machines]
                    total_sum = sum(random_values)
                    
                    if total_sum > 0:
                        normalized_ratios = [val / total_sum for val in random_values]
                        for i, key_tuple in enumerate(key_list_for_machines):
                            # 유효한 위치(mijt>0)에서만 변이가 일어나도록 할 수도 있음
                            if self.mijt.get(key_tuple, 0) > 0:
                                mutated_chromosome[key_tuple] = normalized_ratios[i]
                            else: # 생산 불가 위치는 0으로 유지
                                mutated_chromosome[key_tuple] = 0
                    else: # 모든 랜덤값이 0이면 비율도 0
                        for key_tuple in key_list_for_machines:
                            mutated_chromosome[key_tuple] = 0

                    # 정규화 후에는 해당 그룹 내 기계들의 비율 합이 1.0이 되므로, 다시 한번 정규화 필요
                    current_group_sum = sum(mutated_chromosome.get(key, 0) for key in key_list_for_machines)
                    if current_group_sum > 0:
                        for key_tuple in key_list_for_machines:
                            mutated_chromosome[key_tuple] /= current_group_sum
        
        return mutated_chromosome
        
    def solve(self):
        """
        유전 알고리즘의 전체 최적화 과정을 실행하고,
        최적해, 최적 점수, 그리고 실행 로그를 반환합니다.
        """
        # 1. 초기 모집단 생성
        #    _generate_initial_solution은 이미 내부에서 _decode를 호출함.
        self.population = [self._generate_initial_solution() for _ in range(self.n_pop)]
        # print(f"  [solve] 세대 0: 초기 모집단 생성됨 (크기: {len(self.population)})")
        # print(f"  [solve] 세대 0: 단일 염색체 크기: {len(self._generate_initial_solution())})")
        # 2. 초기 최고해 및 최고 점수 설정
        #    초기 모집단이 비어있을 경우에 대한 방어 코드 추가 가능
        if not self.population:
            print("Warning: 초기 모집단 생성에 실패했거나 비어있습니다.")
            return None, float('inf'), [], []

        # 초기 모집단에서 가장 좋은 해를 찾아 best_solution, best_score로 설정
        # (또는 첫번째 해를 기준으로 초기화 후 루프에서 비교)
        # 여기서는 첫 번째 해를 기준으로 초기화하고, 첫 세대 평가 시 업데이트되도록 함.
        # 이미 _generate_initial_solution이 decode된 해를 반환하므로, 바로 objective 계산 가능.
        # 하지만 _calculate_objective_value 내부에서 _check_constraints를 다시 하므로,
        # _generate_initial_solution에서 반환된 것이 decode만 된 상태여도 괜찮음.
        
        # 첫번째 해를 기준으로 초기 best 설정
        # self.best_solution = self.population[0].copy() # .copy()는 선택 사항, 아래 루프에서 더 좋은 해로 바뀔 것
        # 초기 best_score 계산을 위해 첫번째 해를 평가
        # (주의: _calculate_objective_value는 decode된 해를 받지만, 내부에서 check_constraints를 함)
        # decoded_initial_best_for_score = self.population[0] # _generate_initial_solution이 decode된 해 반환 가정
        # self.best_score = self._calculate_objective_value(decoded_initial_best_for_score)
        
        # 좀 더 명확하게 하기 위해: 첫 세대 평가 후 best를 결정하는 것이 좋음.
        # 여기서는 일단 초기값을 설정하고, 첫 세대 루프에서 실제 best를 찾도록 함.
        self.best_solution = self.population[0].copy() # 깊은 복사로 초기 설정
        self.best_score = float('inf') # 실제 점수는 첫 세대 평가 후 업데이트

        # 3. 로그 초기화
        self.log = [] # [(세대, 최고점수), ...]
        self.log_detail = [] # [(세대, 최고해_딕셔너리, 최고점수), ...]
        
        no_improvement_streak = 0

        # 4. 메인 GA 루프 시작 (tqdm으로 진행률 표시 가능)
        # from tqdm import tqdm # 만약 사용한다면 파일 상단에 import
        # for gen in tqdm(range(self.n_iter), desc="GA Generations"):
        for gen in range(self.n_iter):
            # print(f"\n  [solve] --- 세대 {gen + 1} 시작 ---")
            
            # 4a. 현 모집단의 각 해에 대한 적합도(점수) 계산
            #    _calculate_objective_value는 내부적으로 _check_constraints를 호출함
            scores = [self._calculate_objective_value(p.copy()) for p in self.population] # p.copy()로 원본 보호
            # print(f"    [solve] 세대 {gen + 1}: 점수 계산 완료 - Scores: {scores}")
            
            # 4b. 현재 세대의 최고해를 전체 최고해와 비교하여 업데이트
            current_gen_min_score = min(scores)
            current_gen_best_idx = scores.index(current_gen_min_score)
            
            if current_gen_min_score < self.best_score:
                self.best_score = current_gen_min_score
                self.best_solution = self.population[current_gen_best_idx].copy() # 깊은 복사
                no_improvement_streak = 0
                # TODO: 원래 코드의 '>best! {gen}, {scores[i]}' 로그를 여기서 찍어줄 수 있음
                # print(f'>best! generation {gen+1}, score {self.best_score}')
            else:
                no_improvement_streak += 1
            
            self.log.append((gen + 1, self.best_score))
            self.log_detail.append((gen + 1, self.best_solution.copy(), self.best_score)) # 필요시 주석 해제

            # 4c. 조기 종료 조건 확인
            if no_improvement_streak >= 200: # 이 값은 하이퍼파라미터로 뺄 수도 있음
                # print(f"세대 {gen+1}에서 조기 종료: {no_improvement_streak} 세대 동안 개선 없음.")
                break
                
            # 4d. 선택 연산 (헬퍼 메소드 호출)
            selected_population = self._perform_selection(scores)
            # print(f"    [solve] 세대 {gen + 1}: 선택 연산 후 모집단 크기: {len(selected_population)}")
            
            # 4e. 교배 및 변이 연산으로 다음 세대 자식들 생성
            children = []
            # selected_population을 순회하며 짝을 지어 교배 및 변이
            # (self.n_pop이 홀수일 경우 마지막 하나는 처리 안 될 수 있으므로, 짝수 크기 가정 또는 보완 로직 필요)
            for i in range(0, len(selected_population), 2): # selected_population 크기만큼
                if i + 1 < len(selected_population): # 부모 쌍이 있는지 확인
                    parent1 = selected_population[i]
                    parent2 = selected_population[i+1]
                    
                    child1, child2 = self._perform_crossover(parent1, parent2) # 교배 메소드 호출
                    
                    child1 = self._apply_mutation(child1) # 변이 메소드 호출
                    child2 = self._apply_mutation(child2)
                    
                    children.append(child1)
                    children.append(child2)
                # else: # 만약 선택된 부모가 홀수 개라서 마지막 하나가 남는다면
                #     if i < len(selected_population):
                #         # 변이만 적용해서 추가하거나, 그대로 추가하거나 등의 전략 사용 가능
                #         children.append(self._apply_mutation(selected_population[i]))
            # print(f"    [solve] 세대 {gen + 1}: 자식 세대 생성 완료 (자식 수: {len(children)})")   

            # 4f. 다음 세대 모집단 구성 (모집단 크기 self.n_pop 유지)
            if not children and self.population : # 만약 자식이 하나도 안 만들어졌다면 (극단적 경우)
                # print("Warning: 자식 해가 생성되지 않았습니다. 이전 세대 모집단을 유지합니다.")
                # self.population = [p.copy() for p in self.population] # 이전 세대 유지 (변화 없음)
                pass # 또는 다른 전략. 여기서는 그냥 두면 다음 세대도 동일한 population으로 진행될 수 있음.
            elif children:
                # 자식 수가 부족하면 선택된 부모 중에서 (또는 다른 방식으로) 채워서 n_pop 유지
                if len(children) < self.n_pop:
                    needed = self.n_pop - len(children)
                    # 예시: selected_population에서 부족한 만큼 추가 (이미 .copy()된 객체들임)
                    # 또는, 이전 세대의 best_solution을 추가하는 엘리트 전략 등
                    if selected_population: # selected_population이 비어있지 않다면
                         # selected_population 에서 랜덤하게 또는 점수 순으로 추가 가능
                        for k in range(min(needed, len(selected_population))):
                            children.append(selected_population[k].copy()) # 예시로 앞에서부터 채움

                self.population = children[:self.n_pop] # n_pop 크기로 자름
                # print(f"    [solve] 세대 {gen + 1}: 다음 세대 모집단 구성 완료 (크기: {len(self.population)})")
                # print(f"  [solve] --- 세대 {gen + 1} 종료 (Best Score: {self.best_score}) ---")
                
                # 만약 그래도 모집단 크기가 부족하다면 (children도 selected_population도 부족했다면)
                while len(self.population) < self.n_pop:
                    # print("Warning: 모집단 크기를 맞추기 위해 새로운 랜덤 해를 추가 생성합니다.")
                    self.population.append(self._generate_initial_solution())
            # else: children도 없고 self.population도 없는 경우는 초기화 실패로 이미 반환했을 것.



        # 5. 최종 최고해, 최고 점수, 로그 반환
        return self.best_solution, self.best_score, self.log, self.log_detail