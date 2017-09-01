# Interference Aware Cluster Management


## Background

### [Relevant papers](https://github.com/xiandong79/Interference-Aware-Cluster-Management/tree/master/papers)
1. Paragon: QoS-Aware Scheduling for Heterogeneous Datacenters - ASPLOS ’13
2. Quasar: Resource-Efficient and QoS-Aware Cluster Management - ASPLOS ’14

## [Testbed Installation](https://github.com/xiandong79/Interference-Aware-Cluster-Management/tree/master/installation-docs)

1. SPEC CPU2006 单线程负载
2. memcached 内存型数据库
3. parsec 多线程负载
4. websearch Latency critical 任务
5. perf/lib-perf 任务性能检测

## [Test & collect data](https://github.com/xiandong79/Interference-Aware-Cluster-Management/tree/master/shell-scripts)

1. [bash_basic.sh](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/shell-scripts/bash_basic.sh) - SPEC CPU 2006 任务间相互干扰
2. [memcached+spec2006.sh](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/shell-scripts/memcached%2Bspec2006.sh) - memcached+spec2006 任务间相互干扰
3. [ibench.sh](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/shell-scripts/ibench.sh) [autorun_ibench.sh](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/shell-scripts/autorun_ibench.sh) - SPEC CPU 2006 任务在 ibench 七种不同压力干扰下的运行状况

## Data Analysis

1. [ALS_SGD_MF.py](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/data-analysis-scripts/ALS_SGD_MF.py) 

```
# Train rmse: 0.632234683903
# Test rmse: 0.958863923627
```
2. [gridsearch_ALS_SGD_MF.py](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/data-analysis-scripts/gridsearch_ALS_SGD_MF.py) - 可遍历地求出最优超参数

3. [基于服务器 IPS参数 的推荐系统-20170823版本.html](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/data-analysis-scripts/基于服务器%20IPS参数%20的推荐系统-20170823版本.html)

```
载入原始数据¶
In [2]:
# Load data from disk

names = ['workload_id', 'pressure_id', 'rating']
df = pd.read_csv('/Users/dong/Desktop/体系-数据分析/IPS-rating-final.csv',delimiter=",", names=names)

print(df.shape)

num_workloads = df.workload_id.unique().shape[0]
num_pressures = df.pressure_id.unique().shape[0]

print(num_workloads, "kinds of workloads")
print(num_pressures, "kinds of pressures")
(86, 3)
12 kinds of workloads
8 kinds of pressures
```



## Results and Conclusion
在未来的使用中，每次任务提交时，只需在IPS-rating-final.csv文件中，继续补充 此种workload_id 的在 pressure_id 测试值（2-3次），即可得出此种 workload在每一种压力下的 “百分制评分”。 Greedily选择最高评分即可。

```
Prediction Result...
[[ 53  81  52  74 100  96  99 101]
 [ 49  79  49  72  98  94  97 100]
 [ 50  79  48  71  99  95  98 100]
 [ 51  78  47  70  97  93  97  99]
 [ 50  78  49  71  97  94  97  99]
 [ 50  78  50  71  96  93  95  98]
 [ 51  78  47  70  97  93  97  99]
 [ 86  81  58  76  88  87  92  95]
 [ 49  80  51  73 100  96  99 101]
 [ 50  79  50  72  98  94  97 100]
 [ 69  80  70  77  87  85  87  88]
 [ 79  80  78  79  82  81  82  83]]
``` 
 

![](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/pictures/libquantum_CPI.PNG)

![](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/pictures/libquantum_IPS.PNG)

![](https://github.com/xiandong79/Interference-Aware-Cluster-Management/blob/master/pictures/sjeng_IPS.PNG)