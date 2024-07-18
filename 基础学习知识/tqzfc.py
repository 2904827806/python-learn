class Rect:
    def __init__(self,w,h):
        self.w = w
        self.h = h
    @property
    def area(self):
        return self.w*self.h
reac = Rect(800,60)
a =reac.area


