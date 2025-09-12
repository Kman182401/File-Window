#!/usr/bin/env python3
"""
Real-time resource monitoring for trading pipeline
Tracks memory, CPU, disk usage during system execution
"""
import psutil
import time
import json
import threading
import os
from datetime import datetime

class ResourceMonitor:
    def __init__(self, log_file="resource_usage.log", alert_memory_gb=5.0):
        self.log_file = log_file
        self.alert_memory_gb = alert_memory_gb
        self.monitoring = False
        self.data = []
        
    def get_metrics(self):
        """Get current system metrics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/home/ubuntu')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get Python process memory if available
        try:
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                if 'python' in proc.info['name'].lower():
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024
                    })
        except:
            python_processes = []
            
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'memory_total_gb': memory.total / 1024**3,
            'memory_used_gb': memory.used / 1024**3,
            'memory_available_gb': memory.available / 1024**3,
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'disk_used_gb': disk.used / 1024**3,
            'disk_free_gb': disk.free / 1024**3,
            'disk_percent': (disk.used / disk.total) * 100,
            'python_processes': python_processes
        }
        
        # Memory alert
        if metrics['memory_used_gb'] > self.alert_memory_gb:
            print(f"⚠️  MEMORY ALERT: {metrics['memory_used_gb']:.1f}GB used (>{self.alert_memory_gb}GB threshold)")
            
        return metrics
    
    def monitor_continuous(self, interval=5):
        """Continuously monitor resources"""
        self.monitoring = True
        print(f"🔍 Starting continuous monitoring (interval: {interval}s)")
        print(f"📊 Memory alert threshold: {self.alert_memory_gb}GB")
        print(f"📝 Logging to: {self.log_file}")
        print("-" * 80)
        
        while self.monitoring:
            try:
                metrics = self.get_metrics()
                self.data.append(metrics)
                
                # Console output
                print(f"{metrics['timestamp'][11:19]} | "
                      f"MEM: {metrics['memory_used_gb']:.1f}GB/{metrics['memory_total_gb']:.1f}GB "
                      f"({metrics['memory_percent']:.1f}%) | "
                      f"CPU: {metrics['cpu_percent']:.1f}% | "
                      f"DISK: {metrics['disk_percent']:.1f}%")
                
                # Log to file
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                    
                time.sleep(interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                
        print("🛑 Monitoring stopped")
        
    def start_background(self, interval=5):
        """Start monitoring in background thread"""
        self.monitor_thread = threading.Thread(target=self.monitor_continuous, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        
    def get_summary(self):
        """Get summary of monitoring session"""
        if not self.data:
            return "No monitoring data collected"
            
        memory_usage = [d['memory_used_gb'] for d in self.data]
        cpu_usage = [d['cpu_percent'] for d in self.data]
        
        summary = {
            'duration_minutes': len(self.data) * 5 / 60,  # Assuming 5s intervals
            'memory_peak_gb': max(memory_usage),
            'memory_avg_gb': sum(memory_usage) / len(memory_usage),
            'cpu_peak_percent': max(cpu_usage),
            'cpu_avg_percent': sum(cpu_usage) / len(cpu_usage),
            'memory_alerts': sum(1 for m in memory_usage if m > self.alert_memory_gb)
        }
        
        return summary

if __name__ == "__main__":
    monitor = ResourceMonitor()
    try:
        monitor.monitor_continuous()
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()
        print("Summary:", monitor.get_summary())