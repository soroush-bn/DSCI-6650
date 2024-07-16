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

Part two, Question 3: 
```bash
python3 part2.py -q3 --gui  
```
