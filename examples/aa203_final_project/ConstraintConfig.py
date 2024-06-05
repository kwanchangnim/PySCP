class LinkageConfig():
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class ConstraintConfig():
    def __init__(self, isp, fsb, sb, cb, t0b, tfb):
        self.init_state_bound = isp
        self.final_state_bound = fsb
        self.state_bound = sb
        self.control_bound = cb
        self.t0_bound = t0b
        self.tf_bound = tfb
        self.lazy_grow_sb()

    def lazy_grow_sb(self):
        if self.state_bound is not None:
            for i in range(len(self.state_bound[0])):
                minimum = min(self.init_state_bound[0,i], self.state_bound[0,i], self.final_state_bound[0,i])
                maximum = max(self.init_state_bound[1,i], self.state_bound[1,i], self.final_state_bound[1,i])
                self.state_bound[0,i] = minimum
                self.state_bound[1,i] = maximum