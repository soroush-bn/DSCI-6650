# DSCI-6650
Reinforcement learning course - Spring 2024

## Project one:

Clone the project:
```bash
git clone https://github.com/soroush-bn/DSCI-6650.git
```

Reproducing Results:

On Ubuntu, navigate to the project directory:
```bash
cd /DSCI-6650/project1/
```
Run the following command to reproduce results for Part 1:
```bash
python3 part1.py --steps 1000 --num_problems 1000 --save True

```

You can also customize alpha and epsilon values using the following options:

    --alpha to specify the alpha value
    --epsilon to specify the epsilon value


Run the following command to reproduce results for Part 2:
```bash
python3 part2.py --steps 10000 --num_problems 1000 --save True --non_stationary "gradual" --gradual_type "drift"
python3 part2.py --steps 10000 --num_problems 1000 --save True --non_stationary "gradual" --gradual_type "revert"
python3 part2.py --steps 10000 --num_problems 1000 --save True --non_stationary "abrupt"
```



## Project Two:


Reproducing Results:


```bash
cd /DSCI-6650/project2/
```


Part one, Question 1: 
```bash
python3 part1.py -q1 --gui  
```

Part one, Question 2: 
```bash
python3 part1.py -q2 --gui  
```

Part two, Question 1: 
```bash
python3 part2.py -q1 --gui  
```

Part two, Question 2: 
```bash
python3 part2.py -q2 --gui  
```


if you want add params tuning you can add --tune flag (if applicable to that question), for changing the params under tuning, please refeer to code
Part two, Question 3: 
```bash
python3 part2.py -q3 --gui  
```



## Project Three:
Reproducing Results:


```bash
cd /DSCI-6650/project3/
```


Part one:
experiment 1:
```bash
python3 part1.py --episodes 500 --alpha 0.1 --epsilon 0.2
```
experiment 2:
```bash
python3 part1.py --episodes 1000 --alpha 0.1 --epsilon 0.2
```
experiment 3:
```bash
python3 part1.py --episodes 5000 --alpha 0.1 --epsilon 0.2
```
experiment 4:
```bash
python3 part1.py --episodes 10000 --alpha 0.1 --epsilon 0.2
```

Part two:

```bash
python3 part1.py --episodes 10000 --alpha 0.1 
```
experiment 2:
```bash
python3 part1.py --episodes 10000 --alpha 0.2 
```
experiment 3:
```bash
python3 part1.py --episodes 10000 --alpha 0.3 
```
experiment 4:
```bash
python3 part1.py --episodes 10000 --alpha 0.5 
```
