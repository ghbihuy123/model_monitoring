# ML_observasion
## Cấu trúc thư mục
```python
|__ assets/
|..|__ data_asset/ #chứa các dữ liệu ví dụ
|..|__ model_aset/ #chứa mô hình ví dụ
|__ assettypes/ # folder sinh ra khi kết nối đến icp4d
|__ dist/ #chứa file wheel là file đã được đóng gói thành package
|__ examples/ # các file ví dụ
|...|__ example.py # ví dụ sử dụng về tính toán data drift, model quality
|...|__ memory_example.py # ví dụ sử dụng memory profiling
|__ logs/ # lưu ví dụ output log của memory profiling
|__ notebooks/
|...|__ main.ipynb #Notebook ví dụ cho tính toán data drift, model quality
|...|__ memprofile.ipynb # #Notebook ví dụ cho memory profiling
|...|__ observability_2.ipynb # #Notebook draft xây thư viện
|...|__ train_model.ipynb # Notebook ví dụ train mô hình
|__ wheels/ # Lưu các thư viện cần thiết
```

## Yêu cầu thư viện

- `memory_profiler>=0.61.0`
- `numpy>=2.1.2`
- `pandas>=2.2.3`
- `scikit_learn>=1.5.2`
- `scipy>=1.14.1`
- `shap>=0.46.0`

## Build package

- xóa folder `build` (nếu có)

- chạy lệnh sau để build package

```bash
python setup.py sdist bdist_wheel
```

- sau khi build thành công, file wheels sẽ có tại folder `dist`



## Hướng dẫn sử dụng

Có 2 cách thức sử dụng thư viện

**Cách 1:** Copy folder `ml_obs` vào dự án bạn đang làm

**Cách 2:** Cài bằng pip install 

Để cài đặt package ml_obs ta thực hiện lệnh sau

Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

```bash
pip install dist/ml_obs-0.0.1-py3-none-any.whl
```

### Import các package cần thiết

```python
import json
import pandas as pd
from ml_obs.utils.data_utils import Dataset, ObsConfig
from ml_obs.data_quality import DataQuality
from ml_obs.model_quality import ModelQuality
from ml_obs.drift_calculator import DriftCalculator
from ml_obs.explainer import Explainer
from ml_obs.fairness import Fairness
from ml_obs.memprofile import log_memory_usage

```

`ml_obs` hỗ trợ các Components như:
- `DataQuality`: Kiểm tra chất lượng dữ liệu
- `ModelQuality`: Kiểm tra chất lượng mô hình
- `DriftCalculator`: Tính toán các chỉ số liên quan đến drift
- `Explainer`: Giải thích kết quả mô hình
- `Fairness`: Tính fairness của mô hình
- `log_memory_usage`: Tính memory đã sử dụng

### Load dữ liệu

```python
log_train = pd.read_csv('assets/data_asset/inputs/log_training.csv')
log_production = pd.read_csv('assets/data_asset/inputs/log_production.csv')
with open('assets/data_asset/inputs/metadata.json', 'r') as file:
    metadata = json.load(file)
model = joblib.load('assets/model_asset/model.pkl') # có thể thay thể bằng pickle, cách load mô hình phụ thuộc vào cách bạn lưu mô hình, không phụ thuộc vào package ml_obs
```
#### Format dữ liệu đầu vào

file `log_train` và file `log_production` cần thỏa mãn format sau. Trong đó
- `model_type`: phải thuộc kiểu string và chỉ thuộc 1 trong các loại
    - 'Binary'
    - 'Regression'
    - 'Forecasting'
    - 'Multiclass'
    - 'Ranking'
Có thể tham khảo tại  `ml_obs/utils/data_utils.py` để xem các config về các loại model_type

|Trên trường| Ý nghĩa| Ví dụ, ghi chú|
|--|--|--|
|**sampleID**|ID của phần tử||
|**model_code**| mã mô hình|
|**model_version**| version mô hình|
|**model_type**| loại bài toán| Binary, Multiclass, Ranking, Regression, Forecasting|
|**environment**| môi trường của dữ liệu| Training, Validation, Development, Production|
|**timestamp**| thời gian thu thập dữ liệu||
|feature x_1| biến đầu vào của mô hình | |
|feature x_2| biến đầu vào của mô hình | |
|...|||
|feature x_n| biến đầu vào của mô hình | |
|**actual_label**|nhãn thực tế|Một nhãn duy nhất nếu là classification hoặc regression. Nếu là ranking thì trả danh sách các items theo thứ tự score giảm dần|
|**actual_prob**| xác suất của nhãn thực|Trong trường hợp hợp là Regression thì để trường này bằng Null. Trong trường hợp Ranking thì trả về danh sách score của từng items|
|**prediction_label**|nhãn dự đoán| Một nhãn duy nhất nếu là classification hoặc regression. Nếu là ranking thì trả danh sách các items theo thứ tự score giảm dần|
|**prediction_prob**|xác suất dự đoán nhãn|Trong trường hợp hợp là Regression thì trường này mang giá trị Null|
|**pos_label**| nhãn dương tính|Áp dụng cho binary classification, các trường hợp khác mang giá trị Null|
|**threshold_label**| ngưỡng cắt|Chỉ áp dụng cho binary classification trả ra nhãn|

> Lưu ý: các trường được bôi đậm bắt buộc phải đúng tên trường

#### Format của file `metadata.json`


|Trên trường| Ý nghĩa| Ví dụ, ghi chú|
|--|--|--|
|**model_code**| mã mô hình|
|**model_version**| version mô hình|
|**numerical_features**| danh sách các features dạng số||
|**categorical_features**| danh sách các features dạng phân loại||
|**segment_features**| danh sách các features cần kiểm tra model quality theo phân cấp||
|**fairness_features**| danh sách các features cần theo dõi tính fairness||
|**explainability_features**| danh sách các features cần giải thích||
|**pos_label**| nhãn dương tính|Áp dụng cho binary classification, các trường hợp khác mang giá trị Null|
|**threshold_label**| ngưỡng cắt|Chỉ áp dụng cho binary classification trả ra nhãn|
|**class_mapping_cat_to_index**| mảng đối chiếu từ class sang dạng index| Lưu ý nhãn pos_label nên để index cao nhất(vì thư viện SHAP không hỗ trợ phần pos_label cho việc giải thích mô hình mà mặc định theo model, do đó model cũng nên để pos_label ở index cao nhất)

Ví dụ

```json
{"model_code": "CREDIT_LGBM", 
"model_version": "v1", 
"model_type": "Binary", 
"numerical_features": ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"], 
"categorical_features": ["checking_status", "credit_history", "purpose", "savings_status", "employment", "gender", "personal_status", "other_parties", "property_magnitude", "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"], 
"segment_features": ["gender"], 
"fairness_features": ["gender"], 
"explainability_features": ["credit_amount", "duration"], 
"pos_label": "bad", 
"threshold_label": 0.5,
"class_mapping_cat_to_index": {"bad":1, "good":0}}
```

#### Model

`model` có thể lưu và load bằng `pickle` hoặc `joblib` tùy người sử dụng.

### Cấu trúc lại dữ liệu

package `ml_obs` quản lý các dữ liệu đầu vào thông qua các object

- `Dataset` : Object quản lý dữ liệu đầu vào gồm các bản ghi về dữ liệu huấn luyện/đánh giá/ dự đoán và nhãn cùng với một số metadata
- `ObsConfig` : Object quản lý các config cho việc theo dõi mô hình

```python
log_train_dataset = Dataset(log_train)
log_production_dataset = Dataset(log_production)
obs_config = ObsConfig(model_code=metadata['model_code'],
                        model_version=metadata['model_version'],
                        model_type=metadata['model_type'],
                        numerical_features=metadata['numerical_features'],
                        categorical_featues=metadata['categorical_features'],
                        segment_features=metadata['segment_features'],
                        fairness_features=metadata['fairness_features'],
                        explainability_features=metadata['explainability_features'],
                        pos_label=metadata['pos_label'],
                        threshold_label=metadata['threshold_label'],
                        class_mapping_cat_to_index=metadata['class_mapping_cat_to_index'])
```

### Kiểm tra chất lượng dữ liệu

```python
dq = DataQuality(obs_config=obs_config)
report = dq.report(dataset=log_production_dataset)
print(report)
```

### Kiểm tra chất lượng mô hình

```python
mq = ModelQuality(obs_config=obs_config)
report = mq.report(dataset=log_production_dataset)
print(report)
```

### Tính toán chỉ số drift

Yêu cầu hai `Dataset` đầu vào trong đó `current` là tập cần quan sát và `reference` là tập tham chiếu. Thông thường `reference` sẽ là tập liên quan đến tập huấn luyện và tập `current` là tập dữ liệu inference hoặc sử dụng trên production

```python
dc = DriftCalculator(obs_config=obs_config)
report = dc.report(reference=log_train_dataset, current=log_production_dataset)
print(report)
```

### Model Explain

Yêu cầu đầu vào là một tập `Dataset` và `estimator` là mô hình đã được đào tạo

```python
explainer = Explainer(obs_config=obs_config)
report = explainer.report(dataset=log_production_dataset, estimator=model)
#print(report)
```

Đối với output của method partial_dependence có thể thao khảo ở [partial_dependence](https://scikit-learn.org/1.5/modules/generated/sklearn.inspection.partial_dependence.html) của thư viện Scikit-learn

Tài liệu về method SHAP có thể tham khảo tại trang chủ [https://shap.readthedocs.io](https://shap.readthedocs.io/en/latest/index.html)

### Fairness

```python
fairness = Fairness(obs_config=obs_config)
report = fairness.report(dataset=log_production_dataset)
print(report)
```

### Tính dung lượng RAM sử dụng

```python
from ml_obs.memprofile import log_memory_usage

@log_memory_usage(interval=1, timeout=6, logdir='logs/')
def example_function():
    # Một ví dụ hàm
    lst = [i for i in range(10**6)]  # Sử dụng bộ nhớ
    return sum(lst)

# Gọi hàm

if __name__ == '__main__':
    example_function()
```
Trong đó
- `interval` là đơn vị thời gian ghi log theo giây. Mặc định 1 giây
- `timeout` là thời gian chờ kết thúc ghi log cho đến khi hàm `example_function()` kết thúc. Mặc định 6 giây
- `logdir` là thư mục ghi log. Mặc định `''`

Kết quả trả ra sẽ là một file log có tên `example_function_memory.log` và có định dạng

```
Time: 1s, Memory Usage: 47.52 MiB
Time: 2s, Memory Usage: 62.61 MiB
Time: 3s, Memory Usage: 75.41 MiB
Time: 4s, Memory Usage: 85.10 MiB
Time: 5s, Memory Usage: 85.10 MiB
Time: 6s, Memory Usage: 48.45 MiB
```

### Tham khảo

Có thể xem tham khảo tại folder `ml_obs/examples`