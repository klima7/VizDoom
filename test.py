class Test:
    
    def __init__(self) -> None:
        self.a = 0
        
    def __getattr__(self, attr):
        print('getattr', attr)
        
t = Test()
t.b