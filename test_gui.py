import tkinter as tk
from tkinter import ttk, messagebox
import os
from test import test_model
from test_for_different_goal import test_random_images_with_circle_trg
from evacuate import evaluate_model
import threading
import time

class TestController:
    def __init__(self):
        self.test_thread = None
        self.running = False
        self.stop_event = threading.Event()

    def start_test(self, test_method, checkpoint_path, data_dir, max_samples, model_mode, cuda_device, callback):
        if self.running:
            messagebox.showwarning("警告", "已有测试正在运行，请等待完成或停止当前测试")
            return False
        
        self.stop_event.clear()
        self.running = True
        
        self.test_thread = threading.Thread(
            target=self._execute_test,
            args=(test_method, checkpoint_path, data_dir, max_samples, model_mode, cuda_device, callback)
        )
        self.test_thread.daemon = True
        self.test_thread.start()
        return True

    def _execute_test(self, test_method, checkpoint_path, data_dir, max_samples, model_mode, cuda_device, callback):
        try:
            if self.stop_event.is_set():
                return
                
            if test_method == "Circle Target Test":
                test_random_images_with_circle_trg(
                    checkpoint_path, data_dir, 
                    max_samples=max_samples, 
                    modelmode=model_mode, 
                    cuda_device=cuda_device
                )
            elif test_method == "Evacuate Test":
                evaluate_model(
                    checkpoint_path, data_dir, 
                    max_samples=max_samples, 
                    modelmode=model_mode, 
                    cuda_device=cuda_device
                )
            elif test_method == "Test":
                test_model(
                    checkpoint_path, data_dir, 
                    max_samples=max_samples, 
                    modelmode=model_mode, 
                    cuda_device=cuda_device
                )
        except Exception as e:
            callback(False, f"测试过程中发生错误:\n{e}")
        finally:
            self.running = False
            callback(True, "测试完成")

    def stop_test(self):
        if self.running and self.test_thread is not None:
            self.stop_event.set()
            self.test_thread.join(timeout=2)
            self.running = False
            return True
        return False

    def is_running(self):
        return self.running

def get_all_checkpoints():
    """获取所有权重文件的路径"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    return [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

def get_last_checkpoint():
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    return checkpoint_path

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Test GUI")
        self.geometry("800x650")
        
        # 测试控制器
        self.test_controller = TestController()
        
        # 初始化UI
        self.setup_ui()
        
        # 状态变量
        self.test_in_progress = False
        
    def setup_ui(self):
        font_large = ("Arial", 20)
        
        # 数据集选择
        tk.Label(self, text="Choose Dataset：", font=font_large).grid(row=0, column=0, padx=10, pady=10)
        self.dataset_var = tk.StringVar(value="Full Data(train)")
        self.dataset_combobox = ttk.Combobox(self, textvariable=self.dataset_var, state="readonly", font=font_large, width=30)
        self.dataset_combobox["values"] = (
            "Full Data(train)", "Full Data(val)", 
            "Small Data(train)", "Small Data(val)",
            "Full Masked Data(train)", "Full Masked Data(val)", 
            "Small Masked Data(train)", "Small Masked Data(val)"
        )
        self.dataset_combobox.grid(row=0, column=1, padx=10, pady=10)
        
        # 测试方式选择
        tk.Label(self, text="Choose Test Method：", font=font_large).grid(row=1, column=0, padx=10, pady=10)
        self.test_method_var = tk.StringVar(value="Test")
        self.test_method_combobox = ttk.Combobox(self, textvariable=self.test_method_var, state="readonly", font=font_large, width=30)
        self.test_method_combobox["values"] = ("Circle Target Test", "Evacuate Test", "Test")
        self.test_method_combobox.grid(row=1, column=1, padx=10, pady=10)
        
        # 权重文件选择
        tk.Label(self, text="Choose Checkpoint：", font=font_large).grid(row=2, column=0, padx=10, pady=10)
        self.checkpoint_var = tk.StringVar(value="Latest Checkpoint")
        self.checkpoint_combobox = ttk.Combobox(self, textvariable=self.checkpoint_var, state="readonly", font=font_large, width=30)
        self.update_checkpoint_list()
        self.checkpoint_combobox.grid(row=2, column=1, padx=10, pady=10)
        
        # 最大样本数选择
        tk.Label(self, text="Max Samples：", font=font_large).grid(row=3, column=0, padx=10, pady=10)
        self.max_samples_var = tk.StringVar(value="256")
        self.max_samples_combobox = ttk.Combobox(self, textvariable=self.max_samples_var, state="readonly", font=font_large, width=30)
        self.max_samples_combobox["values"] = ("16", "256", "None")
        self.max_samples_combobox.grid(row=3, column=1, padx=10, pady=10)
        
        # 模型模式选择
        tk.Label(self, text="Model Mode：", font=font_large).grid(row=4, column=0, padx=10, pady=10)
        self.model_mode_var = tk.StringVar(value="train")
        self.model_mode_combobox = ttk.Combobox(self, textvariable=self.model_mode_var, state="readonly", font=font_large, width=30)
        self.model_mode_combobox["values"] = ("train", "eval")
        self.model_mode_combobox.grid(row=4, column=1, padx=10, pady=10)
        
        # CUDA 设备选择
        tk.Label(self, text="CUDA Device：", font=font_large).grid(row=5, column=0, padx=10, pady=10)
        self.cuda_device_var = tk.StringVar(value="0")
        self.cuda_device_combobox = ttk.Combobox(self, textvariable=self.cuda_device_var, state="readonly", font=font_large, width=30)
        self.cuda_device_combobox["values"] = ("0", "1")
        self.cuda_device_combobox.grid(row=5, column=1, padx=10, pady=10)
        
        # 状态标签
        self.status_var = tk.StringVar(value="Ready!")
        self.status_label = tk.Label(self, textvariable=self.status_var, font=font_large, fg="blue")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=10)
        
        # 按钮框架
        button_frame = tk.Frame(self)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        # 测试按钮
        self.test_button = tk.Button(
            button_frame, text="Start Test", font=font_large, 
            command=self.start_test, width=10
        )
        self.test_button.pack(side=tk.LEFT, padx=10)
        
        # 停止按钮
        self.stop_button = tk.Button(
            button_frame, text="Stop Test", font=font_large, 
            command=self.stop_test, state=tk.DISABLED, width=10
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # 刷新按钮
        self.refresh_button = tk.Button(
            button_frame, text="Refresh", font=font_large, 
            command=self.update_checkpoint_list, width=10
        )
        self.refresh_button.pack(side=tk.LEFT, padx=10)
    
    def update_checkpoint_list(self):
        checkpoints = ["Latest Checkpoint"] + get_all_checkpoints()
        self.checkpoint_combobox["values"] = checkpoints
        messagebox.showinfo("Refreshed", "Checkpoint list refreshed!")
    
    def start_test(self):
        # 获取用户选择的参数
        dataset = self.dataset_var.get()
        test_method = self.test_method_var.get()
        checkpoint = self.checkpoint_var.get()
        max_samples = self.max_samples_var.get()
        model_mode = self.model_mode_var.get()
        cuda_device = self.cuda_device_var.get()

        # 验证输入
        if not dataset or not test_method or not checkpoint:
            messagebox.showerror("错误", "请选择数据集、测试方式和权重文件！")
            return

        try:
            cuda_device = int(cuda_device)
        except ValueError:
            messagebox.showerror("错误", "无效的CUDA设备选择！")
            return

        if max_samples == "None":
            max_samples = None
        else:
            try:
                max_samples = int(max_samples)
            except ValueError:
                messagebox.showerror("错误", "无效的最大样本数选择！")
                return

        # 配置数据集路径
        data_dirs = {
            "Full Data(train)": "filtered_data/all/train",
            "Full Data(val)": "filtered_data/all/val",
            "Small Data(train)": "filtered_data/small_256/train",
            "Small Data(val)": "filtered_data/small_256/val",
            "Full Masked Data(train)": "filtered_data/ground_mask_all/train",
            "Full Masked Data(val)": "filtered_data/ground_mask_all/val",
            "Small Masked Data(train)": "filtered_data/ground_mask_256/train",
            "Small Masked Data(val)": "filtered_data/ground_mask_256/val"
        }
        
        data_dir = data_dirs.get(dataset)
        if data_dir is None:
            messagebox.showerror("错误", "无效的数据集选择！")
            return

        # 获取检查点路径
        try:
            if checkpoint == "Latest Checkpoint":
                checkpoint_path = get_last_checkpoint()
            else:
                checkpoint_path = os.path.join("checkpoints", checkpoint)
        except FileNotFoundError as e:
            messagebox.showerror("错误", f"无法找到检查点文件：\n{e}")
            return

        # 打印测试信息
        print(f"测试方式: {test_method}")
        print(f"权重文件路径: {checkpoint_path}") 
        print(f"数据集路径: {data_dir}")
        print(f"最大样本数: {max_samples}")
        print(f"模型模式: {model_mode}")
        print(f"CUDA设备: {cuda_device}")
        
        # 更新UI状态
        self.status_var.set("Test running...")
        self.test_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # 启动测试线程
        success = self.test_controller.start_test(
            test_method, checkpoint_path, data_dir, 
            max_samples, model_mode, cuda_device,
            self.on_test_complete
        )
        
        if not success:
            self.status_var.set("Ready!")
            self.test_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def stop_test(self):
        if self.test_controller.stop_test():
            self.status_var.set("Stopped")
        else:
            self.status_var.set("No test running")
        
        self.test_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def on_test_complete(self, success, message):
        self.after(0, self._handle_test_complete, success, message)
    
    def _handle_test_complete(self, success, message):
        if success:
            self.status_var.set("Test complete")
            messagebox.showinfo("Finished", message)
        else:
            self.status_var.set("Error")
            messagebox.showerror("Error", message)
        
        self.test_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = Application()
    app.mainloop()