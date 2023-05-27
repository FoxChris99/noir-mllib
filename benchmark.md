+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
|    Linear Regression   | Ordinary Least Squares | Ordinary Least Squares | Mini-batch SGD         | SGD regressor  | Data Loading   |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| Dataset                | Noir 16 local replicas | Sklearn Python         | Noir 16 local replicas | Sklearn Python | Pandas Python  |
| (n° rows, n° features) | Training Time          | Training Time          | Training Time 100 iter | Training Time  | Loading time   |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| 1 milion, 4            |         0.24 s         |          0.1 s         |           1 s          |      1.5 s     |     0.6 s      |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| 10 milion, 4           |         2.7 s          |          1 s           |          9.8 s         |      14.2 s    |     5.5 s      |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| 100 milion, 4          |                        |                        |                        |                |                |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| 1 bilion, 4            |                        |                        |                        |                |                |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| 1 milion, 7            |                        |                        |                        |                |                |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| 10 milion, 7           |                        |                        |                        |                |                |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| 100 milion, 7          |          116 s         |         100 s          |          140 s         |      266 s     |      140 s     |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+
| 10 milion, 20          |                        |                        |                        |                |                |
+------------------------+------------------------+------------------------+------------------------+----------------+----------------+




+------------------------+------------------------+----------------+------------------------+----------------+
|   Logistic Regression  | Adam optimizer         | LBFGS          | Mini-batch SGD         | SGD classifier |
+------------------------+------------------------+----------------+------------------------+----------------+
| Dataset                | Noir 16 local replicas | Sklearn Python | Noir 16 local replicas | Sklearn Python |
| (n° rows, n° features) | Training Time          | Training Time  | Training Time          | Training Time  |
+------------------------+------------------------+----------------+------------------------+----------------+
| 1 milion, 4            |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 10 milion, 4           |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 100 milion, 4          |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 1 bilion, 4            |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 1 milion, 7            |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 10 milion, 7           |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 100 milion, 7          |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 1 bilion, 7            |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+




+------------------------+------------------------+----------------------+
| Support Vector Machine | LinearSVC              | LinearSVC Classifier |
|                        | Binary Classifier      | Binary Classifier    |
+------------------------+------------------------+----------------------+
| Dataset                | Noir 16 local replicas | Sklearn Python       |
| (n° rows, n° features) | Training Time          | Training Time        |
+------------------------+------------------------+----------------------+
| 1 milion, 4            |                        |                      |
+------------------------+------------------------+----------------------+
| 10 milion, 4           |                        |                      |
+------------------------+------------------------+----------------------+
| 100 milion, 4          |                        |                      |
+------------------------+------------------------+----------------------+
| 1 bilion, 4            |                        |                      |
+------------------------+------------------------+----------------------+
| 1 milion, 7            |                        |                      |
+------------------------+------------------------+----------------------+
| 10 milion, 7           |                        |                      |
+------------------------+------------------------+----------------------+
| 100 milion, 7          |                        |                      |
+------------------------+------------------------+----------------------+
| 1 bilion, 7            |                        |                      |
+------------------------+------------------------+----------------------+




+------------------------+------------------------+----------------+------------------------+----------------+
|      Random Forest     | Random Forest          | Random Forest  | Random Forest          | Random Forest  |
|                        | Classifier             | Classifier     | Regressor              | Regressor      |
+------------------------+------------------------+----------------+------------------------+----------------+
| Dataset                | Noir 16 local replicas | Sklearn Python | Noir 16 local replicas | Sklearn Python |
| (n° rows, n° features) | Training Time          | Training Time  | Training TIme          | Training Time  |
+------------------------+------------------------+----------------+------------------------+----------------+
| 1 milion, 4            |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 10 milion, 4           |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 100 milion, 4          |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 1 bilion, 4            |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 1 milion, 7            |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 10 milion, 7           |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 100 milion, 7          |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+
| 1 bilion, 7            |                        |                |                        |                |
+------------------------+------------------------+----------------+------------------------+----------------+