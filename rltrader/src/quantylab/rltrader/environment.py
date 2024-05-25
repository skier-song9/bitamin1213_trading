class Environment:
    PRICE_IDX = 4  # 종가의 위치 ⚠️수정 필요

    def __init__(self, chart_data=None):
        self.chart_data = chart_data # 주식 종목의 차트 데이터 -> ndim = 2
        self.observation = None # 현재 관측치
        self.idx = -1 # chart_data에서 현재 위치를 나타내는 index

    def reset(self):
        # episode마다 observation과 idx를 초기한다.
        self.observation = None
        self.idx = -1

    def observe(self):
        # idx를 다음 위치로 이동시키고 observation을 업데이트한다.
        if len(self.chart_data) > self.idx + 1: # chart_data의 끝까지 이동한다
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        # 현재 observation에서 "종가"를 확인한다. ⚠️수정 필요? no 필요시 PRICE_INDEX만 수정.
        if self.observation is not None:
            return self.observation.iloc[self.PRICE_IDX]
        return None
