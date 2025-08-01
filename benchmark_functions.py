import numpy as np
from typing import Tuple, Callable

class BenchmarkFunctions:
    """
    توابع معیار برای تست الگوریتم‌های بهینه‌سازی
    معادل F1-F23 در پیاده‌سازی MATLAB
    """
    
    @staticmethod
    def get_function_details(func_name: str) -> Tuple[np.ndarray, np.ndarray, int, Callable]:
        """
        دریافت جزئیات تابع شامل مرزها، بعد و تابع هدف
        
        Args:
            func_name: نام تابع (F1, F2, ..., F23)
            
        Returns:
            تاپل شامل (مرزهای پایینی، مرزهای بالایی، بعد، تابع هدف)
        """
        func_map = {
            'F1': (BenchmarkFunctions.f1, -100, 100, 10),
            'F2': (BenchmarkFunctions.f2, -10, 10, 10),
            'F3': (BenchmarkFunctions.f3, -100, 100, 10),
            'F4': (BenchmarkFunctions.f4, -100, 100, 10),
            'F5': (BenchmarkFunctions.f5, -30, 30, 10),
            'F6': (BenchmarkFunctions.f6, -100, 100, 10),
            'F7': (BenchmarkFunctions.f7, -1.28, 1.28, 10),
            'F8': (BenchmarkFunctions.f8, -500, 500, 10),
            'F9': (BenchmarkFunctions.f9, -5.12, 5.12, 10),
            'F10': (BenchmarkFunctions.f10, -32, 32, 10),
            'F11': (BenchmarkFunctions.f11, -600, 600, 10),
            'F12': (BenchmarkFunctions.f12, -50, 50, 10),
            'F13': (BenchmarkFunctions.f13, -50, 50, 10),
            'F14': (BenchmarkFunctions.f14, -65.536, 65.536, 2),
            'F15': (BenchmarkFunctions.f15, -5, 5, 4),
            'F16': (BenchmarkFunctions.f16, -5, 5, 2),
            'F17': (BenchmarkFunctions.f17, [-5, 0], [10, 15], 2),
            'F18': (BenchmarkFunctions.f18, -2, 2, 2),
            'F19': (BenchmarkFunctions.f19, 0, 1, 3),
            'F20': (BenchmarkFunctions.f20, 0, 1, 6),
            'F21': (BenchmarkFunctions.f21, 0, 10, 4),
            'F22': (BenchmarkFunctions.f22, 0, 10, 4),
            'F23': (BenchmarkFunctions.f23, 0, 10, 4)
        }
        
        if func_name not in func_map:
            raise ValueError(f"Function {func_name} not found. Available functions: {list(func_map.keys())}")
        
        func, lb, ub, dim = func_map[func_name]
        
        if isinstance(lb, (int, float)):
            lb = np.array([lb] * dim)
        if isinstance(ub, (int, float)):
            ub = np.array([ub] * dim)
            
        return lb, ub, dim, func
    
    @staticmethod
    def f1(x):
        """F1: تابع کره - تابع تک‌قله‌ای ساده"""
        return np.sum(x**2)
    
    @staticmethod
    def f2(x):
        """F2: مجموع مقادیر مطلق و حاصل‌ضرب"""
        return np.sum(np.abs(x)) + np.prod(np.abs(x))
    
    @staticmethod
    def f3(x):
        """F3: مجموع مربعات مجموع‌های جزئی"""
        dim = len(x)
        result = 0
        for i in range(dim):
            result += np.sum(x[:i+1])**2
        return result
    
    @staticmethod
    def f4(x):
        """F4: حداکثر مقدار مطلق"""
        return np.max(np.abs(x))
    
    @staticmethod
    def f5(x):
        """F5: تابع روزنبراک - تابع تک‌قله‌ای دشوار"""
        dim = len(x)
        return np.sum(100 * (x[1:dim] - x[0:dim-1]**2)**2 + (x[0:dim-1] - 1)**2)
    
    @staticmethod
    def f6(x):
        """F6: Step function"""
        return np.sum(np.abs(x + 0.5)**2)
    
    @staticmethod
    def f7(x):
        """F7: Quartic function with noise"""
        dim = len(x)
        return np.sum(np.arange(1, dim + 1) * x**4) + np.random.random()
    
    @staticmethod
    def f8(x):
        """F8: Schwefel function"""
        return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def f9(x):
        """F9: Rastrigin function"""
        dim = len(x)
        return np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim
    
    @staticmethod
    def f10(x):
        """F10: Ackley function"""
        dim = len(x)
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - 
                np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1))
    
    @staticmethod
    def f11(x):
        """F11: Griewank function"""
        dim = len(x)
        return (np.sum(x**2) / 4000 - 
                np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1)
    
    @staticmethod
    def f12(x):
        """F12: Penalized function 1"""
        dim = len(x)
        y = 1 + (x + 1) / 4
        
        term1 = 10 * np.sin(np.pi * y[0])**2
        term2 = np.sum(((y[:-1] - 1)**2) * (1 + 10 * np.sin(np.pi * y[1:])**2))
        term3 = (y[-1] - 1)**2
        penalty = np.sum(BenchmarkFunctions._u_function(x, 10, 100, 4))
        
        return (np.pi / dim) * (term1 + term2 + term3) + penalty
    
    @staticmethod
    def f13(x):
        """F13: Penalized function 2"""
        dim = len(x)
        
        term1 = np.sin(3 * np.pi * x[0])**2
        term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
        term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
        penalty = np.sum(BenchmarkFunctions._u_function(x, 5, 100, 4))
        
        return 0.1 * (term1 + term2 + term3) + penalty
    
    @staticmethod
    def f14(x):
        """F14: Shekel's Foxholes function"""
        a = np.array([[-32, -16, 0, 16, 32] * 5,
                      [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5])
        
        result = 0
        for j in range(25):
            b = np.sum((x - a[:, j])**6)
            result += 1 / (j + 1 + b)
        
        return (1 / 500 + result)**(-1)
    
    @staticmethod
    def f15(x):
        """F15: Kowalik function"""
        a = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 
                      0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        b = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
        
        return np.sum((a - (x[0] * (b**2 + x[1] * b)) / 
                      (b**2 + x[2] * b + x[3]))**2)
    
    @staticmethod
    def f16(x):
        """F16: Six-Hump Camel-Back function"""
        return (4 * x[0]**2 - 2.1 * x[0]**4 + x[0]**6 / 3 + 
                x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4)
    
    @staticmethod
    def f17(x):
        """F17: Branin function"""
        return ((x[1] - 5.1 * x[0]**2 / (4 * np.pi**2) + 5 * x[0] / np.pi - 6)**2 + 
                10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)
    
    @staticmethod
    def f18(x):
        """F18: Goldstein-Price function"""
        return ((1 + (x[0] + x[1] + 1)**2 * 
                (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * 
                (30 + (2 * x[0] - 3 * x[1])**2 * 
                (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2)))
    
    @staticmethod
    def f19(x):
        """F19: Hartman's 3-Dimensional function"""
        a = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        c = np.array([1, 1.2, 3, 3.2])
        p = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], 
                      [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
        
        result = 0
        for i in range(4):
            result -= c[i] * np.exp(-np.sum(a[i, :] * (x - p[i, :])**2))
        
        return result
    
    @staticmethod
    def f20(x):
        """F20: Hartman's 6-Dimensional function"""
        a = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
        c = np.array([1, 1.2, 3, 3.2])
        p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                      [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                      [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                      [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
        
        result = 0
        for i in range(4):
            result -= c[i] * np.exp(-np.sum(a[i, :] * (x - p[i, :])**2))
        
        return result
    
    @staticmethod
    def f21(x):
        """F21: Shekel's 5-Dimensional function"""
        a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                      [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        
        result = 0
        for i in range(5):
            result -= 1 / (np.sum((x - a[i, :])**2) + c[i])
        
        return result
    
    @staticmethod
    def f22(x):
        """F22: Shekel's 7-Dimensional function"""
        a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                      [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        
        result = 0
        for i in range(7):
            result -= 1 / (np.sum((x - a[i, :])**2) + c[i])
        
        return result
    
    @staticmethod
    def f23(x):
        """F23: Shekel's 10-Dimensional function"""
        a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                      [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        
        result = 0
        for i in range(10):
            result -= 1 / (np.sum((x - a[i, :])**2) + c[i])
        
        return result
    
    @staticmethod
    def _u_function(x, a, k, m):
        """Helper function for penalty terms"""
        return k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a) 