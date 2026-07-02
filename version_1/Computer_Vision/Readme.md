#### $$IoU (Intersection Over Union)=\frac{Area of Overlap/Intersection}{Area of union}$$

#### Confusion Matrix  
|Metric|Prediction|Actual Reality|Definition|Note|
|-------|----------|-------------|----------|----------|
|TP|True (Positive)|True (Positive)|Model correctly identified a positive case.|Success
|TN|False (Negative)|False (Negative)|Model correctly identified a negative case.|Success
|FP|True (Positive)|False (Negative)|Model said Yes, but it was actually No.|Type I Error (False Alarm)
|FN|False (Negative)|True (Positive)|Model said No, but it was actually Yes.|Type II Error (Missing Opportunity) 

#### $$Precision=\frac{TP}{TP+FP}$$
#### $$Recall=\frac{TP}{TP+FN}$$
#### $$Accuracy=\frac{TP}{TP+FP+FN+TN}$$


#### Dos & Don't

1. Atleast HD image
2. Avoid blur image
3. Avoid cluttery background image