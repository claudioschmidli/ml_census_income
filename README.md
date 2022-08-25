# Adult Income Dataset a.k.a "Census Income"
### _Advanced Machine Learning FHNW Brugg_
Claudio Schmidli

Date: 25.08.2022
### Task
Predict whether income exceeds $50K/yr based on census data. Also known as [Census Income" dataset](https://www.kaggle.com/datasets/rdcmdev/adult-income-dataset?select=adult.data).

#### My code is splitted into the following parts:
- Exploratory Data Analysis (<a href="EDA.ipynb">EDA.ipynb</a>)
- Investigation of feature importance (<a href="Feature_Importance.ipynb">Feature_Importance.ipynb</a>)
- Training of a logistic regression model (<a href="Logistic_regression.ipynb">Logistic_regression.ipynb</a>)
- Training of a boosted trees model (<a href="lightgbm.ipynb">lightgbm.ipynb</a>)
- Training of a model using AutoML (<a href="AutoML.ipynb">AutoML.ipynb</a>)
- Functions and classes for data processing (<a href="modules/preprocess.py">modules/preprocess.py</a>)
- Functions and classes for data visualizations (<a href="modules/plot.py">modules/plot.py</a>)

### Assessment criteria of this project
I fullfilled the expected assessment criteria as follows:
<table>
<thead>
  <tr>
    <th>Criteria</th>
    <th>Implementation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>richtiges Anwenden der Besonderheiten des data-sets</td>
    <td>
      <ul>
          <li>Different missing value treatments were tested (dropping, most freq. imputation, predict missing)</li>
          <li>Class imbalance was handled using SMOTE and model class weights</li>
          <li>Features with low permuation importance were excluded</li>
          <li>The redundant column education was removed because education_num contains the same in a ordered format</li>
          <li>(see notebooks <a href="Logistic_regression.ipynb">Logistic_regression.ipynb</a>/<a href="lightgbm.ipynb">lightgbm.ipynb</a>)</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>Auswahl richtige loss-Funktion</td>
    <td>
      <ul>
          <li>I used the f1 score to evaluate the model</li>
          <li>This makes sense because we have imbalanced data</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>Verteilung der target-Values</td>
    <td>
      <ul>
          <li>I visualized the impalanced targets (see <a href="EDA.ipynb">EDA.ipynb</a>)</li>
          <li>I treated the impalanced data with SMOTE and class weights (see <a href="Logistic_regression.ipynb">Logistic_regression.ipynb</a>/<a href="lightgbm.ipynb">lightgbm.ipynb</a>)</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>feature-engineering</td>
    <td>
      <ul>
          <li>I tested interactions for the Logistic regression and lightgbm classifiers (see <a href="Logistic_regression.ipynb">Logistic_regression.ipynb</a>/<a href="lightgbm.ipynb">lightgbm.ipynb</a>) </li>
          <li>I created a custom transformer to generate the interactions (see <a href="modules/preprocess.py">modules/preprocess.py</a>)</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>korrekte Einschätzung der Modell-Performance</td>
    <td>
      <ul>
          <li>I used cross validation to estimate the performance of the model</li>
          <li>I prevented data leakage by using pipelines and creating custom transformers (see <a href="modules/preprocess.py">modules/preprocess.py</a>)</li>
          <li>I used the imblearn library to do SMOTE withing the cross validation</li>
          <li>I used StratifiedKFold for cross validation</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>richtiges Anwenden von feature-importance</td>
    <td>
      <ul>
          <li>I used premuatation improtance to estimate the performance of a feature</li>
          <li>I created a transformer to create interactions based on the best features (see InteractionsTransformer in <a href="modules/preprocess.py">modules/preprocess.py</a>)</li>
          <li>I visualized the feature importance in  <a href="Feature_Importance.ipynb">Feature_Importance.ipynb</a></li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>grid-search: korrekte Definition des Such-Raums</td>
    <td>
      <ul>
          <li>I used grid search to optimize model parameters (see <a href="Logisticd_regression.ipynb">Logistic_regression.ipynb</a>/<a href="lightgbm.ipynb">lightgbm.ipynb</a>)</li>
          <li>I started with the default values</li>
          <li>I also used non linear search patterns</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>Verwendung von Pipelines</td>
    <td>
      <ul>
          <li>I used pipelines to generate and evaluate my models (see <a href="Logistic_regression.ipynb">Logistic_regression.ipynb</a>/<a href="lightgbm.ipynb">lightgbm.ipynb</a>)</li>
          <li>I created custom transformers to process data (see <a href="modules/preprocess.py">modules/preprocess.py</a>)</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>Suche nach bestem Algorithmus (Lern-Algorithmus); <br>mindestens 2 verschieden Ansätze</td>
    <td>
      <ul>
          <li>I trained a logistic regression model (see <a href="Logistic_regression.ipynb">Logistic_regression.ipynb</a>)</li>
          <li>I trained a gradient boosted tree (see <a href="lightgbm.ipynb">lightgbm.ipynb</a>)</li>
          <li>I trained a model using the AutoML toolkit <a href="https://auto.gluon.ai/dev/tutorials/tabular_prediction/tabular-custom-metric.html">AutoGluon</a> (see <a href="AutoML.ipynb">AutoML.ipynb</a>)</li>
    </ul>
    </td>
  </tr>
</tbody>
</table>

# Summary

As expected the boosted trees provided the best results:
### <a href="Logistic_regression.ipynb">Logistic_regression</a>
- f1 score train: 0.68897
- f1 score test: 0.67885

### <a href="lightgbm.ipynb">lightgbm</a>
- f1 score train: 0.72795
- f1 score test: 0.71947

### <a href="AutoML.ipynb">AutoGluon</a>
Best model (WeightedEnsemble_L2)
- f1 score train: 0.73069
- f1 score test: 0.71536

Second best model (LightGBM_BAG_L1)
- f1 score train: 0.72576
- f1 score test: 0.69749

### Conclusion

- Gradient Boosting models such as LightGBM, XGBoost and Catboost have long been considered best in class for tabular data. The results here confrim this statement.
- Fortunately I got a better test core with my own lightgbm mode than the AutoML model of AutoGluon :)

### Question
- How can I achive such high scores as shown in the leaderbord of [this](https://www.kaggle.com/competitions/adult-census-income/data?select=AdultCensusIncomeTest.csv) Kaggle competition?
