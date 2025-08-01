#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اسکریپت تست جامع توابع معیار الگوریتم سنجاقک
Comprehensive Benchmark Testing Script for Dragonfly Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dragonfly_algorithm import DragonflyAlgorithm
from benchmark_functions import BenchmarkFunctions

def test_single_function(func_name, search_agents=40, max_iterations=500, verbose=True):
    """
    تست یک تابع معیار خاص
    
    Args:
        func_name: نام تابع (F1, F2, ..., F23)
        search_agents: تعداد سنجاقک‌ها
        max_iterations: حداکثر تعداد تکرارها
        verbose: نمایش جزئیات
    
    Returns:
        dict: نتایج تست شامل امتیاز، موقعیت، منحنی همگرایی و زمان اجرا
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"تست تابع {func_name}")
        print(f"{'='*60}")
    
    # دریافت جزئیات تابع
    lb, ub, dim, fobj = BenchmarkFunctions.get_function_details(func_name)
    
    if verbose:
        print(f"بعد مسئله: {dim}")
        print(f"مرزهای پایینی: {lb}")
        print(f"مرزهای بالایی: {ub}")
        print(f"تعداد سنجاقک‌ها: {search_agents}")
        print(f"حداکثر تکرارها: {max_iterations}")
    
    # ایجاد نمونه الگوریتم
    da = DragonflyAlgorithm(search_agents, max_iterations)
    
    # اجرای بهینه‌سازی و اندازه‌گیری زمان
    start_time = time.time()
    best_score, best_pos, convergence_curve = da.optimize(fobj, dim, lb, ub)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # تحلیل نتایج
    initial_score = convergence_curve[0]
    final_score = convergence_curve[-1]
    improvement = initial_score - final_score
    
    # تعیین کیفیت همگرایی
    if final_score < 1e-6:
        convergence_quality = "عالی"
    elif final_score < 1e-3:
        convergence_quality = "خوب"
    elif final_score < 1e-1:
        convergence_quality = "متوسط"
    else:
        convergence_quality = "ضعیف"
    
    results = {
        'function_name': func_name,
        'best_score': best_score,
        'best_position': best_pos,
        'convergence_curve': convergence_curve,
        'execution_time': execution_time,
        'dimension': dim,
        'initial_score': initial_score,
        'final_score': final_score,
        'improvement': improvement,
        'convergence_quality': convergence_quality
    }
    
    if verbose:
        print(f"\nنتایج:")
        print(f"  بهترین امتیاز: {best_score:.8f}")
        print(f"  بهترین موقعیت: {best_pos}")
        print(f"  زمان اجرا: {execution_time:.3f} ثانیه")
        print(f"  کیفیت همگرایی: {convergence_quality}")
        print(f"  بهبود کلی: {improvement:.8f}")
    
    return results

def test_multiple_functions(function_list, search_agents=40, max_iterations=500):
    """
    تست چندین تابع معیار
    
    Args:
        function_list: لیست نام توابع
        search_agents: تعداد سنجاقک‌ها
        max_iterations: حداکثر تعداد تکرارها
    
    Returns:
        dict: نتایج تمام تست‌ها
    """
    
    print(f"\n{'='*80}")
    print(f"تست {len(function_list)} تابع معیار")
    print(f"{'='*80}")
    
    all_results = {}
    
    for func_name in function_list:
        results = test_single_function(func_name, search_agents, max_iterations, verbose=True)
        all_results[func_name] = results
    
    return all_results

def compare_functions(function_list, search_agents=40, max_iterations=500):
    """
    مقایسه همگرایی توابع مختلف
    
    Args:
        function_list: لیست نام توابع
        search_agents: تعداد سنجاقک‌ها
        max_iterations: حداکثر تعداد تکرارها
    """
    
    print(f"\n{'='*80}")
    print(f"مقایسه همگرایی {len(function_list)} تابع")
    print(f"{'='*80}")
    
    # تست تمام توابع
    all_results = test_multiple_functions(function_list, search_agents, max_iterations, verbose=False)
    
    # رسم مقایسه
    plt.figure(figsize=(15, 10))
    
    # نمودار منحنی‌های همگرایی
    plt.subplot(2, 2, 1)
    for func_name, results in all_results.items():
        plt.semilogy(results['convergence_curve'], label=func_name, linewidth=2)
    plt.title('مقایسه منحنی‌های همگرایی', fontsize=14, fontweight='bold')
    plt.xlabel('تکرار')
    plt.ylabel('بهترین امتیاز (مقیاس لگاریتمی)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # نمودار بهترین امتیازها
    plt.subplot(2, 2, 2)
    func_names = list(all_results.keys())
    best_scores = [all_results[func]['best_score'] for func in func_names]
    colors = ['green' if score < 1e-3 else 'orange' if score < 1e-1 else 'red' for score in best_scores]
    
    bars = plt.bar(func_names, best_scores, color=colors, alpha=0.7)
    plt.title('بهترین امتیازهای نهایی', fontsize=14, fontweight='bold')
    plt.ylabel('بهترین امتیاز')
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # اضافه کردن مقادیر روی نمودار
    for bar, score in zip(bars, best_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{score:.2e}', ha='center', va='bottom', fontsize=8)
    
    # نمودار زمان‌های اجرا
    plt.subplot(2, 2, 3)
    execution_times = [all_results[func]['execution_time'] for func in func_names]
    plt.bar(func_names, execution_times, color='skyblue', alpha=0.7)
    plt.title('زمان‌های اجرا', fontsize=14, fontweight='bold')
    plt.ylabel('زمان (ثانیه)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # نمودار کیفیت همگرایی
    plt.subplot(2, 2, 4)
    quality_scores = []
    quality_labels = []
    for func_name in func_names:
        quality = all_results[func_name]['convergence_quality']
        if quality == "عالی":
            score = 4
        elif quality == "خوب":
            score = 3
        elif quality == "متوسط":
            score = 2
        else:
            score = 1
        quality_scores.append(score)
        quality_labels.append(f"{func_name}\n({quality})")
    
    colors = ['green' if score == 4 else 'lightgreen' if score == 3 else 'orange' if score == 2 else 'red' for score in quality_scores]
    plt.bar(range(len(quality_scores)), quality_scores, color=colors, alpha=0.7)
    plt.title('کیفیت همگرایی', fontsize=14, fontweight='bold')
    plt.ylabel('کیفیت (1=ضعیف تا 4=عالی)')
    plt.xticks(range(len(quality_scores)), quality_labels, rotation=45, ha='right')
    plt.ylim(0, 4.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # نمایش خلاصه نتایج
    print(f"\n{'='*80}")
    print(f"خلاصه نتایج")
    print(f"{'='*80}")
    print(f"{'تابع':<10} {'بهترین امتیاز':<15} {'زمان (ث)':<10} {'کیفیت':<10}")
    print(f"{'-'*80}")
    
    for func_name in func_names:
        results = all_results[func_name]
        print(f"{func_name:<10} {results['best_score']:<15.2e} {results['execution_time']:<10.3f} {results['convergence_quality']:<10}")

def parameter_study(function_name='F1', population_sizes=[20, 40, 60, 80], max_iterations=300):
    """
    مطالعه تأثیر پارامترها بر عملکرد الگوریتم
    
    Args:
        function_name: نام تابع برای مطالعه
        population_sizes: لیست اندازه‌های جمعیت
        max_iterations: حداکثر تعداد تکرارها
    """
    
    print(f"\n{'='*80}")
    print(f"مطالعه پارامترها - تابع {function_name}")
    print(f"{'='*80}")
    
    # دریافت جزئیات تابع
    lb, ub, dim, fobj = BenchmarkFunctions.get_function_details(function_name)
    
    results = {}
    
    for pop_size in population_sizes:
        print(f"\nتست با {pop_size} سنجاقک...")
        
        da = DragonflyAlgorithm(pop_size, max_iterations)
        start_time = time.time()
        best_score, best_pos, convergence_curve = da.optimize(fobj, dim, lb, ub)
        end_time = time.time()
        
        results[f'{pop_size} سنجاقک'] = {
            'best_score': best_score,
            'execution_time': end_time - start_time,
            'convergence_curve': convergence_curve
        }
    
    # رسم نتایج
    plt.figure(figsize=(15, 5))
    
    # نمودار منحنی‌های همگرایی
    plt.subplot(1, 3, 1)
    for label, data in results.items():
        plt.semilogy(data['convergence_curve'], label=label, linewidth=2)
    plt.title(f'تأثیر اندازه جمعیت - {function_name}', fontsize=14, fontweight='bold')
    plt.xlabel('تکرار')
    plt.ylabel('بهترین امتیاز')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # نمودار بهترین امتیازها
    plt.subplot(1, 3, 2)
    labels = list(results.keys())
    best_scores = [results[label]['best_score'] for label in labels]
    plt.bar(labels, best_scores, color='lightcoral', alpha=0.7)
    plt.title('بهترین امتیازهای نهایی', fontsize=14, fontweight='bold')
    plt.ylabel('بهترین امتیاز')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # نمودار زمان‌های اجرا
    plt.subplot(1, 3, 3)
    execution_times = [results[label]['execution_time'] for label in labels]
    plt.bar(labels, execution_times, color='lightblue', alpha=0.7)
    plt.title('زمان‌های اجرا', fontsize=14, fontweight='bold')
    plt.ylabel('زمان (ثانیه)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # نمایش خلاصه
    print(f"\n{'='*80}")
    print(f"خلاصه مطالعه پارامترها")
    print(f"{'='*80}")
    print(f"{'اندازه جمعیت':<15} {'بهترین امتیاز':<15} {'زمان (ث)':<10}")
    print(f"{'-'*80}")
    
    for label in labels:
        data = results[label]
        print(f"{label:<15} {data['best_score']:<15.2e} {data['execution_time']:<10.3f}")

def main():
    """
    تابع اصلی برای اجرای تمام تست‌ها
    """
    
    print("🧪 تست جامع الگوریتم سنجاقک")
    print("="*50)
    
    # تست توابع اصلی
    main_functions = ['F1', 'F5', 'F9', 'F10', 'F11']
    
    # 1. تست تک تک توابع
    print("\n1️⃣ تست تک تک توابع:")
    for func in main_functions:
        test_single_function(func, search_agents=40, max_iterations=500)
    
    # 2. مقایسه توابع
    print("\n2️⃣ مقایسه توابع:")
    compare_functions(main_functions, search_agents=40, max_iterations=500)
    
    # 3. مطالعه پارامترها
    print("\n3️⃣ مطالعه پارامترها:")
    parameter_study('F1', population_sizes=[20, 40, 60, 80], max_iterations=300)
    
    print("\n✅ تمام تست‌ها با موفقیت انجام شد!")

if __name__ == "__main__":
    main() 