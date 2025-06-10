# Battle of the Bots

**SOC-2025**  
Mentored by **Sandeep Reddy Nallamilli** and **Yash Sabale**

This repository contains weekly modules, exercises, and learning resources for the SOC. Please remember to fork this repository before starting. It is recommended to use a Linux-based system, preferably Ubuntu or WSL for those on windows.


## Setup Instructions

To set up the project environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Weekly Tasks

### [Week 0](./Week0/)
**Topic:** PyTorch Basics  
**Focus:**
- Manual and `nn.Sequential` neural network implementation in PyTorch
- Introduction to backpropagation and training loops

### [Week 1](./Week1/)
**Topic:** MultiArmedBandits and Q Learning  
**Focus:**
- Epsilon Greedy (exploration vs exploitation) and Upper confidence bounds
- Q Learning and expected rewards stored in tabulature something

### [Week 2](./Week2/)
**Topic:** Deep Q Networks  
**Focus:**
- Learning how Q Learning fails on large state spaces and how to represent them
- Creating reward functions for different environments
