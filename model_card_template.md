# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest model using 100 estimators and default scikit-learn hyperparameters. Trained with sklearn version 1.3.2.

## Intended Use

For classifying whether income exceeds $50K/yr based on census data, extracted by Barry Becker from the 1994 Census databas

## Training Data

The training data contains 26048 records of census data with 14 features: 'age', 'workclass', 'fnlgt', 'education', 'education-num','marital-status', 'occupation', 'relationship', 'race', 'sex','capital-gain', 'capital-loss', 'hours-per-week', 'native-country'.

## Evaluation Data

The evaluation data contains 6513 records of census data with 14 features: 'age', 'workclass', 'fnlgt', 'education', 'education-num','marital-status', 'occupation', 'relationship', 'race', 'sex','capital-gain', 'capital-loss', 'hours-per-week', 'native-country'.

## Metrics

Precison, recall and F1 score are the performance metrics used to evaluate.
The model achieved a performance of precision of 0.72, recall of 0.63 and F1 score of 0.67.

## Ethical Considerations

None

## Caveats and Recommendations

The average salary level of female record is lower than that of male record, which could be a potential source of bias to make model prone to show a higher prediction on female's income level than it should be, while lower prediction on male's income level than it should be.