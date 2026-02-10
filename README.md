
# Team Number – Project Title

## Team Info
- 22471A05E1 — Adhikari Satish ( [LinkedIn](https://www.linkedin.com/in/adhikari-satish-gpm420?utm_source=share_via&utm_content=profile&utm_medium=member_android) )
_Work Done:Problem identification, dataset collection, preprocessing, model design, ensemble architecture, implementation, results analysis, documentation.
- 22471A05G8 — Tullibilli Lakshmi Siva Sai ( [LinkedIn]((https://www.linkedin.com/in/siva-sai-60b510369?utm_source=share_via&utm_content=profile&utm_medium=member_android)) )
_Work Done: Literature survey, exploratory data analysis (EDA), feature analysis, model training support, evaluation metrics

- 22471A05I1 — Pallapu Harish ( [LinkedIn](https://www.linkedin.com/in/harish-pallapu-215990363?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) )
_Work Done: esting, result validation, visualization, deployment preparation, report formatting.


---

## Abstract
Water quality plays a vital role in protecting public health, agriculture, and ecosystems, yet real-time monitoring remains a challenge due to irregular sampling, regional variations, and the limitations of traditional prediction models. To address these gaps, this paper introduces AquaNet-X, a novel deep hybrid ensemble model designed for accurate and scalable Water Quality Index (WQI) prediction. AquaNet-X integrates Bidirectional GRU for sequential dynamics, Transformer layers for capturing long-range feature dependencies, and boosting algorithms (XGBoost and LightGBM) for nonlinear tabular interactions, all unified through a Meta-CatBoost stacked learner. This architecture balances the strengths of deep learning and ensemble methods, reducing variance while enhancing interpretability and robustness. This experiment was conducted using a real-world Indian surface water quality dataset with multivariate parameters such as pH, DO, BOD, and temperature, preprocessed into supervised sequences. The proposed model achieving 99.94% prediction accuracy, thereby setting a new state-of-the-art benchmark, significantly outperforming existing baselines. The novelty of AquaNet-X lies in its meta-layered hybridization strategy, which enables cross-regional adaptability, real-time deployment, and reliable generalization across diverse water sources. It is better way to predict water quality index in different regions based on multivariate features. AquaNet-X is a next-generation tool for intelligent water quality monitoring and sustainable water governance

---

## Paper Reference (Inspiration)
👉 **[Paper Title A Hybrid Deep Learning Model for Water Quality Prediction]
  – Author Names Sivaratri Siva Nageswara Rao1, Adhikari Satish2, Tullibilli Lakshmi Siva Sai3, Pallapu Harish4, Mallikarjuna Rao Gundavarapu5, Patri Venkata Sesha Sudha Arundathi Parimala6, Dodda Venkatarededy7
 (https://doi.org/10.1109/ACCESS.2025.3580741)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Introduced a hybrid ensemble architecture combining deep learning and machine learning models

Improved prediction accuracy using stacked ensemble learning

Enhanced preprocessing and normalization techniques

Achieved better generalization across multiple water sources

Optimized performance for real-time prediction use cases

---

## About the Project
This project predicts the Water Quality Index (WQI) from sensor-based water quality parameters.

Workflow:

Input: Raw water quality parameters (pH, DO, BOD, Temperature)

Processing: Data cleaning, normalization, feature engineering

Model: Hybrid deep ensemble (GRU / CNN / ML models)

Output: Predicted Water Quality Index and quality category

The system helps environmental agencies and researchers monitor water quality efficiently and take preventive actions.
---

## Dataset Used
👉 **[GRQA – Indian Surface Water Quality Dataset](https://drive.google.com/file/d/11klegEdomcuh573X8IpgramFwXcyksvy/view?usp=drive_link)**

**Dataset Details:**
Government-sourced water quality records

Multivariate time-series data

Covers multiple regions and time periods

Reliable for academic and research use

---

## Dependencies Used
Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, TensorFlow, Keras, XGBoost, LightGBM
---

## EDA & Preprocessing
Missing value handling

Outlier detection

Min-Max normalization

Feature correlation analysis

Supervised learning data framing

---

## Model Training Info
Hybrid ensemble model

Combination of deep learning and ML predictors

Trained using historical water quality records

Optimized using performance-based loss metrics

---

## Model Testing / Evaluation
Metrics used: RMSE, MAE, R² Score

Cross-validation performed

Model evaluated on unseen test data

---

## Results
High prediction accuracy

Improved R² score compared to baseline models

Stable performance across datasets

Efficient real-time prediction capability

---

## Limitations & Future Work
Limited to selected physicochemical parameters

Can be extended with IoT-based real-time sensors

Future scope includes mobile/web deployment and larger datasets

---

## Deployment Info
Model ready for integration with IoT water sensors

Can be deployed using Flask / FastAPI

Suitable for cloud-based monitoring platforms

---
