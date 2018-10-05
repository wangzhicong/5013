# This is Kuojian Lu's branch

-----

## LSTM model
### how
- predict the probability of price up and down in the next time step
- predict the increaing rate of price in the next time step
### problem
- loss converges with difficulty
- low accuracy

## strategy
### how
- simply compare the predicted probability or increaing rate with a threshold to decide to buy, sell or hold on
### problem
- may always sell a bitcoin at every time step, thus stopping too early