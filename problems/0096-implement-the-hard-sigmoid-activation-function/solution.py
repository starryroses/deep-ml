def hard_sigmoid(x: float) -> float:
    if x<= -2.5:
        return 0
    elif x<2.5 and x>-2.5:
        return 0.2*x + 0.5
    elif x>= 2.5:
        return 1