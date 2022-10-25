class EarlyStopping:
    """
    Perform early stopping based on the user-defined metric
    given a tolerance.
    """

    def __init__(self, metric="loss", tolerance=5, mode="min"):
        """
        Args:
            metric (str): Validation metric to perform early stop on
            tolerance (int): How long to wait until validation metric improves
            mode (str): Whether validation metric should be minimized or maximized
        """
        self.metric = metric
        self.tolerance = tolerance
        assert mode in ["min", "max"]
        self.mode = mode

        self.best_score = None
        self.counter = 0

    def __call__(self, validation_stats):
        assert self.metric in validation_stats

        stop = False
        cur_score = validation_stats[self.metric]

        if self.best_score is None:
            self.best_score = cur_score
        elif self.mode == "min":
            if cur_score < self.best_score:
                self.best_score = cur_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if cur_score > self.best_score:
                self.best_score = cur_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            stop = True

        return stop
