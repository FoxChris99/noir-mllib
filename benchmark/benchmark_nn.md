

MLP with ADAM (dataset easy 100 epoch)
+------------------------+------------------------+------------------------+------------------------+------------------------+
|  Neural Network 3k par |  Noir Regr full batch  | Noir Regr 0.1 batch    | Noir Regr decentralized|  Noir Regr full batch  |
+------------------------+------------------------+------------------------+------------------------+------------------------+
| Dataset                | Noir 16 local replicas | Noir 16 local replicas | Noir 16 local replicas | Noir 1 local replica   |
| (n° rows, n° features) | Training Time 100 iter | Training Time 1000 iter| Training Time 100 iter | Training Time 100 iter | 
+------------------------+------------------------+------------------------+------------------------+------------------------+
| 100 thousand, 10       |          20.7 s (33%)  |         23.1 s (85%)   |        33.4 s (87%)    |         139s (33%)     |     
+------------------------+------------------------+------------------------+------------------------+------------------------+
noir with 0.01 batch -> 61s 92%
noir with 0.001 batch -> 380s 91%

+------------------------+------------------------+------------------------+------------------------+
|  Neural Network 3k par |  Tensorflow full batch | Tensorflow batch 1000  | Tensorflow batch 100   |
+------------------------+------------------------+------------------------+------------------------+
| Dataset                | Regression             | Regression             | Regression             |
| (n° rows, n° features) | Training Time 100 iter | Training Time 100 iter | Training Time 100 iter |
+------------------------+------------------------+------------------------+------------------------+
| 100 thousand, 10       |          6 s           |         14 s           |        82 s            |     
+------------------------+------------------------+------------------------+------------------------+

+------------------------+------------------------+------------------------+------------------------+------------------------+
|  Neural Network 3k par |  Noir Class full batch | Noir Class 0.2 batch   | Sklearn MLP full batch | Sklearn MLP 0.2 batch  |
+------------------------+------------------------+------------------------+------------------------+------------------------+
| Dataset                | Noir 16 local replicas | Noir 16 local replicas | Classification         | Classification         |
| (n° rows, n° features) | Training Time 100 iter | Training Time 1000 iter| Training Time 100 iter | Training Time 100 iter |
+------------------------+------------------------+------------------------+------------------------+------------------------+
| 100 thousand, 10       |          23 s          |           25 s         |          37 s          |         27 s           |    
+------------------------+------------------------+------------------------+------------------------+------------------------+



dataset harder to converge -> 1000 epoch, early stopping: tol 1e-5, patience 20
+------------------------+------------------------+----------------------------+----------------------------+
|  Neural Network 3k par |  Noir Class 0.1 batch  | Tensorflow Class 0.1 batch |   Sklearn Class 0.1 batch  |
+------------------------+------------------------+----------------------------+----------------------------+
| Dataset                | Noir 16 local replicas | Noir 16 local replicas     | Noir 16 local replicas     |
| (n° rows, n° features) | Stop 199 epoch         | Stop   511 epoch           | Stop   237 epoch           |
+------------------------+------------------------+----------------------------+----------------------------+
| 100 thousand, 10       |        55 s (84%)      |     30 s (87%)             |             59 s (85%)     |
+------------------------+------------------------+----------------------------+----------------------------+



100 epoch with a larger network. tensorflow 40x faster, sklearn 4x faster
+------------------------+------------------------+----------------+
|  Neural Network 37k par| Noir Regr 0.1 batch    | MLP Regr tensor|
+------------------------+------------------------+----------------+
| Dataset                | Noir 16 local 37k par  | Tensorflow cpu |
| (n° rows, n° features) | Training Time 1000 iter| Python         |
+------------------------+------------------------+----------------+
| 100 thousand, 10       |       526 s            |       18 s     |
+------------------------+------------------------+----------------+



