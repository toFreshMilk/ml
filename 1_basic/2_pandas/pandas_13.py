# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import pyplot as plt

# pandas DataFrame의 시각화 기능
# pandas는 내부적으로 matplotlib를 활용하여
# 시각화를 제공할 수 있도록 구현되어 있음.

data = {
    'year': [2016, 2017, 2018],
    'GDP rate': [2.8, 3.1, 3.0],
    'GDP': ['1.637M', '1.73M', '1.83M']
}
 
df = pd.DataFrame(data)

# DataFrame의 hist 메소드를 사용하여 
# 각 데이터의 히스토그램(빈도)을 확인할 수 있음
df.hist()
plt.show()

# DataFrame의 plot 메소드를 사용하여 시각화를 할 수 있음
df.plot(x='year', y='GDP rate')
plt.show()

# DataFrame의 각 열에 대해서 plot 메소드를 사용하여 
# 시각화를 할 수 있음
# kind 속성에 원하는 차트 형태를 지정할 수 있음
df['GDP rate'].plot(kind='bar', rot=0)
plt.show()

df['GDP rate'].plot()
plt.show()








