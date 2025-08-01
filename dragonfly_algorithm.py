import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import math

class DragonflyAlgorithm:
    """
    الگوریتم سنجاقک (DA) - پیاده‌سازی پایتون
    الگوریتم بهینه‌سازی فراابتکاری الهام گرفته از رفتار گروهی سنجاقک‌ها
    
    مرجع: S. Mirjalili, Dragonfly algorithm: a new meta-heuristic
    optimization technique for solving single-objective, discrete, and 
    multi-objective problems, Neural Computing and Applications
    """
    
    def __init__(self, search_agents_no: int = 40, max_iteration: int = 500):
        """
        سازنده کلاس - مقداردهی اولیه الگوریتم سنجاقک
        
        Args:
            search_agents_no: تعداد عامل‌های جستجو (سنجاقک‌ها)
            max_iteration: حداکثر تعداد تکرارها
        """
        self.search_agents_no = search_agents_no
        self.max_iteration = max_iteration
        
    def initialization(self, dim: int, ub: np.ndarray, lb: np.ndarray) -> np.ndarray:
        """
        مقداردهی اولیه جمعیت سنجاقک‌ها - ایجاد موقعیت‌های تصادفی اولیه
        
        Args:
            dim: بعد مسئله (تعداد متغیرها)
            ub: مرزهای بالایی
            lb: مرزهای پایینی
            
        Returns:
            ماتریس موقعیت‌های اولیه
        """
        if ub.size == 1:
            ub = np.ones(dim) * ub
            lb = np.ones(dim) * lb
            
        positions = np.zeros((self.search_agents_no, dim))
        
        for i in range(dim):
            ub_i = ub[i] if ub.size > 1 else ub[0]
            lb_i = lb[i] if lb.size > 1 else lb[0]
            positions[:, i] = np.random.uniform(lb_i, ub_i, self.search_agents_no)
            
        return positions.T  # Return as (dim, search_agents_no)
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        محاسبه فاصله اقلیدسی بین دو نقطه - برای یافتن همسایه‌ها
        
        Args:
            a: نقطه اول
            b: نقطه دوم
            
        Returns:
            فاصله اقلیدسی
        """
        return np.sqrt(np.sum((a - b) ** 2, axis=0))
    
    def levy(self, d: int) -> np.ndarray:
        """
        تولید گام پرواز لوی - برای اکتشاف زمانی که همسایه‌ای یافت نمی‌شود
        
        Args:
            d: بعد مسئله
            
        Returns:
            گام پرواز لوی
        """
        beta = 3/2
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        
        u = np.random.normal(0, sigma, d)
        v = np.random.normal(0, 1, d)
        step = u / np.abs(v) ** (1 / beta)
        
        return 0.01 * step
    
    def optimize(self, fobj: Callable, dim: int, lb: np.ndarray, ub: np.ndarray) -> Tuple[float, np.ndarray, List[float]]:
        """
        اجرای الگوریتم سنجاقک برای بهینه‌سازی - تابع اصلی الگوریتم
        
        Args:
            fobj: تابع هدف
            dim: بعد مسئله
            lb: مرزهای پایینی
            ub: مرزهای بالایی
            
        Returns:
            تاپل شامل (بهترین امتیاز، بهترین موقعیت، منحنی همگرایی)
        """
        print("DA is optimizing your problem")
        
        # مقداردهی اولیه پارامترها
        if ub.size == 1:
            ub = np.ones(dim) * ub
            lb = np.ones(dim) * lb
            
        # شعاع اولیه محله سنجاقک‌ها
        r = (ub - lb) / 10
        delta_max = (ub - lb) / 10
        
        # مقداردهی اولیه غذا (بهترین راه‌حل) و دشمن (بدترین راه‌حل)
        food_fitness = float('inf')
        food_pos = np.zeros(dim)
        
        enemy_fitness = float('-inf')
        enemy_pos = np.zeros(dim)
        
        # مقداردهی اولیه موقعیت‌ها و سرعت‌ها
        X = self.initialization(dim, ub, lb)  # (dim, search_agents_no)
        fitness = np.zeros(self.search_agents_no)
        delta_x = self.initialization(dim, ub, lb)  # (dim, search_agents_no)
        
        convergence_curve = []
        
        for iter in range(self.max_iteration):
            # به‌روزرسانی پارامترهای تطبیقی
            r = (ub - lb) / 4 + ((ub - lb) * (iter / self.max_iteration) * 2)
            w = 0.9 - iter * ((0.9 - 0.4) / self.max_iteration)
            
            my_c = 0.1 - iter * ((0.1 - 0) / (self.max_iteration / 2))
            if my_c < 0:
                my_c = 0
                
            # وزن‌های پنج رفتار گروهی سنجاقک‌ها
            s = 2 * np.random.random() * my_c  # وزن جداسازی
            a = 2 * np.random.random() * my_c  # وزن هم‌راستایی
            c = 2 * np.random.random() * my_c  # وزن انسجام
            f = 2 * np.random.random()         # وزن جذب غذا
            e = my_c                           # وزن دوری از دشمن
            
            # محاسبه برازندگی برای تمام سنجاقک‌ها و یافتن بهترین و بدترین
            for i in range(self.search_agents_no):
                fitness[i] = fobj(X[:, i])
                
                # به‌روزرسانی بهترین راه‌حل (غذا)
                if fitness[i] < food_fitness:
                    food_fitness = fitness[i]
                    food_pos = X[:, i].copy()
                    
                # به‌روزرسانی بدترین راه‌حل (دشمن)
                if fitness[i] > enemy_fitness:
                    if np.all(X[:, i] < ub) and np.all(X[:, i] > lb):
                        enemy_fitness = fitness[i]
                        enemy_pos = X[:, i].copy()
            
            # به‌روزرسانی موقعیت‌ها
            for i in range(self.search_agents_no):
                # یافتن همسایه‌ها در شعاع محله
                neighbors_delta_x = []
                neighbors_x = []
                
                for j in range(self.search_agents_no):
                    dist = self.distance(X[:, i], X[:, j])
                    if np.all(dist <= r) and np.all(dist != 0):
                        neighbors_delta_x.append(delta_x[:, j])
                        neighbors_x.append(X[:, j])
                
                neighbors_no = len(neighbors_x)
                
                # محاسبه رفتار جداسازی - دوری از همسایه‌ها
                S = np.zeros(dim)
                if neighbors_no > 1:
                    for k in range(neighbors_no):
                        S += (neighbors_x[k] - X[:, i])
                    S = -S
                
                # محاسبه رفتار هم‌راستایی - هم‌راستا شدن با همسایه‌ها
                if neighbors_no > 1:
                    A = np.mean(neighbors_delta_x, axis=0)
                else:
                    A = delta_x[:, i]
                
                # محاسبه رفتار انسجام - حرکت به سمت مرکز همسایه‌ها
                if neighbors_no > 1:
                    C_temp = np.mean(neighbors_x, axis=0)
                else:
                    C_temp = X[:, i]
                C = C_temp - X[:, i]
                
                # محاسبه جذب غذا - حرکت به سمت بهترین راه‌حل
                dist_to_food = self.distance(X[:, i], food_pos)
                if np.all(dist_to_food <= r):
                    F = food_pos - X[:, i]
                else:
                    F = np.zeros(dim)
                
                # محاسبه دوری از دشمن - دوری از بدترین راه‌حل
                dist_to_enemy = self.distance(X[:, i], enemy_pos)
                if np.all(dist_to_enemy <= r):
                    Enemy = enemy_pos + X[:, i]
                else:
                    Enemy = np.zeros(dim)
                
                # مدیریت مرزها - اطمینان از قرارگیری در محدوده مجاز
                for tt in range(dim):
                    if X[tt, i] > ub[tt]:
                        X[tt, i] = lb[tt]
                        delta_x[tt, i] = np.random.random()
                    if X[tt, i] < lb[tt]:
                        X[tt, i] = ub[tt]
                        delta_x[tt, i] = np.random.random()
                
                # به‌روزرسانی موقعیت بر اساس فاصله از غذا
                if np.any(dist_to_food > r):
                    if neighbors_no > 1:
                        # حالت گروهی - استفاده از رفتارهای گروهی
                        for j in range(dim):
                            delta_x[j, i] = (w * delta_x[j, i] + 
                                            np.random.random() * A[j] + 
                                            np.random.random() * C[j] + 
                                            np.random.random() * S[j])
                            
                            if delta_x[j, i] > delta_max[j]:
                                delta_x[j, i] = delta_max[j]
                            if delta_x[j, i] < -delta_max[j]:
                                delta_x[j, i] = -delta_max[j]
                                
                            X[j, i] += delta_x[j, i]
                    else:
                        # پرواز لوی - اکتشاف زمانی که همسایه‌ای وجود ندارد
                        X[:, i] += self.levy(dim) * X[:, i]
                        delta_x[:, i] = 0
                else:
                    # حالت نزدیک به غذا - استفاده از تمام رفتارها
                    for j in range(dim):
                        delta_x[j, i] = (a * A[j] + c * C[j] + s * S[j] + 
                                        f * F[j] + e * Enemy[j]) + w * delta_x[j, i]
                        
                        if delta_x[j, i] > delta_max[j]:
                            delta_x[j, i] = delta_max[j]
                        if delta_x[j, i] < -delta_max[j]:
                            delta_x[j, i] = -delta_max[j]
                            
                        X[j, i] += delta_x[j, i]
                
                # مدیریت نهایی مرزها - اطمینان از قرارگیری در محدوده مجاز
                flag_ub = X[:, i] > ub
                flag_lb = X[:, i] < lb
                X[:, i] = (X[:, i] * (~(flag_ub + flag_lb)) + 
                           ub * flag_ub + lb * flag_lb)
            
            # به‌روزرسانی بهترین راه‌حل و ذخیره منحنی همگرایی
            best_score = food_fitness
            best_pos = food_pos
            convergence_curve.append(best_score)
        
        return best_score, best_pos, convergence_curve

# توابع معیار (معادل F1-F23 در MATLAB)
def sphere_function(x):
    """F1: تابع کره - تابع تک‌قله‌ای ساده"""
    return np.sum(x**2)

def rosenbrock_function(x):
    """F5: تابع روزنبراک - تابع تک‌قله‌ای دشوار"""
    dim = len(x)
    return np.sum(100 * (x[1:dim] - x[0:dim-1]**2)**2 + (x[0:dim-1] - 1)**2)

def rastrigin_function(x):
    """F9: تابع راستریگین - تابع چندقله‌ای با نویز"""
    dim = len(x)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim

def ackley_function(x):
    """F10: تابع اکلای - تابع چندقله‌ای با سطح صاف"""
    dim = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - 
            np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1))

def griewank_function(x):
    """F11: تابع گریوانک - تابع چندقله‌ای با قله‌های منظم"""
    dim = len(x)
    return (np.sum(x**2) / 4000 - 
            np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1)

# مثال استفاده و نمایش نتایج
def main():
    """تابع اصلی برای نمایش عملکرد الگوریتم سنجاقک"""
    
    # پارامترهای الگوریتم
    search_agents_no = 40
    max_iteration = 500
    dim = 10
    
    # تابع تست (تابع کره)
    fobj = sphere_function
    lb = np.array([-100] * dim)
    ub = np.array([100] * dim)
    
    # ایجاد نمونه الگوریتم سنجاقک
    da = DragonflyAlgorithm(search_agents_no, max_iteration)
    
    # اجرای بهینه‌سازی
    best_score, best_pos, convergence_curve = da.optimize(fobj, dim, lb, ub)
    
    # نمایش نتایج
    print(f"The best solution obtained by DA is: {best_pos}")
    print(f"The best optimal value of the objective function found by DA is: {best_score}")
    
    # رسم منحنی همگرایی
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(convergence_curve, 'r-', linewidth=2)
    plt.title('منحنی همگرایی')
    plt.xlabel('تکرار')
    plt.ylabel('بهترین امتیاز')
    plt.grid(True)
    
    # رسم موقعیت نهایی (برای نمایش دو بعدی)
    if dim >= 2:
        plt.subplot(1, 2, 2)
        # ایجاد نمودار پراکندگی ساده دو بعدی از موقعیت‌های نهایی
        plt.scatter(best_pos[0], best_pos[1], c='red', s=100, marker='*', label='بهترین موقعیت')
        plt.title('موقعیت بهترین راه‌حل')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 