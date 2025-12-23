import os
import sys
import subprocess
import multiprocessing as mp
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox


def get_base_and_experiments_dir():
    """
    返回工程根目录以及 experiments 目录的绝对路径。
    假设本文件位于工程根目录 `<repo_root>/output_browser.py`。
    """
    # 现在脚本已经移动到项目根目录，因此 base_dir 直接为当前文件所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(base_dir, "experiments")
    return base_dir, experiments_dir


def view_model(path):
    """
    独立进程中运行的模型查看函数。
    使用 Trimesh 打开指定路径的 mesh 文件。
    """
    try:
        import trimesh
    except ImportError as e:
        # 在子进程中打印错误信息即可
        print("[Model Viewer] Failed to import trimesh:", e, file=sys.stderr)
        return

    if not os.path.isfile(path):
        print(f"[Model Viewer] File not found: {path}", file=sys.stderr)
        return

    # 加载网格
    try:
        mesh = trimesh.load(path, force="mesh")
    except BaseException as e:  # 捕获 pyglet/GL 初始化相关异常
        print(f"[Model Viewer] Failed to load mesh: {path} ({e})", file=sys.stderr)
        return

    if mesh is None or getattr(mesh, "is_empty", False):
        print(f"[Model Viewer] Mesh is empty or invalid: {path}", file=sys.stderr)
        return

    # 简单清理，避免奇怪的重复顶点/面
    try:
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicate_faces()
    except BaseException:
        pass

    # Trimesh 的 show 会打开一个交互窗口；在单独进程中调用即可实现多窗口
    window_title = f"Trimesh Viewer - {os.path.basename(path)}"
    try:
        mesh.show(window_title=window_title)
    except TypeError:
        # 旧版本 trimesh 可能不支持 window_title 参数
        mesh.show()


class ModelBrowserApp:
    """
    基于 tkinter 的简单模型浏览器，用于扫描 experiments 目录并通过 Trimesh 查看模型。
    """

    def __init__(self, root):
        self.root = root
        self.root.title("SPLATAM-ORIGINAL Model Browser")
        self.root.geometry("800x600")

        # 路径设置
        self.base_dir, self.experiments_dir = get_base_and_experiments_dir()

        # 保存子进程引用，避免被 GC（可选）
        self.viewer_processes = []
        # 扩展名过滤：默认只显示 .ply
        self.show_ply_var = tk.BooleanVar(value=True)
        self.show_obj_var = tk.BooleanVar(value=False)
        self.show_npz_var = tk.BooleanVar(value=False)
        # .npz 打开方式（静态 / 动态），互斥且至少一个为 True
        self.method_npz_final_var = tk.BooleanVar(value=True)   # 静态查看（final_recon.py）
        self.method_npz_online_var = tk.BooleanVar(value=False)  # 动态查看（online_recon.py）
        # 名称前缀过滤：splat* / mesh*，默认只启用 mesh*
        self.filter_splat_var = tk.BooleanVar(value=False)
        self.filter_mesh_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.scan_models()

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        # 顶层使用垂直方向的两块：上部列表，下部按钮与状态栏
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.columnconfigure(0, weight=1)

        # 上部：Treeview + 滚动条
        frame_list = ttk.Frame(self.root)
        frame_list.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        frame_list.rowconfigure(0, weight=1)
        frame_list.columnconfigure(0, weight=1)

        # 使用树形结构：#0 作为名称列，额外列为大小、最后修改时间和完整路径
        columns = ("size", "mtime", "fullpath")
        self.tree = ttk.Treeview(
            frame_list,
            columns=columns,
            show="tree headings",
            selectmode="browse",
        )

        # #0 列显示名称（文件 / 文件夹）
        self.tree.heading("#0", text="Name")
        self.tree.column("#0", anchor="w", width=500, stretch=True)

        # 附加列：大小、最后修改时间；fullpath 作为隐藏列保存绝对路径
        self.tree.heading("size", text="Size (MB)")
        self.tree.heading("mtime", text="Modified")
        self.tree.heading("fullpath", text="Full Path")

        self.tree.column("size", anchor="e", width=80, stretch=False)
        self.tree.column("mtime", anchor="center", width=140, stretch=False)
        # fullpath 作为隐藏列
        self.tree.column("fullpath", width=0, stretch=False)
        self.tree["displaycolumns"] = ("size", "mtime")

        vsb = ttk.Scrollbar(frame_list, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame_list, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # 双击事件：打开模型
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        # 下部：按钮 + 状态栏
        frame_bottom = ttk.Frame(self.root)
        frame_bottom.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        # 列：0 为左侧控件，1 为状态栏（自适应）
        frame_bottom.columnconfigure(0, weight=0)
        frame_bottom.columnconfigure(1, weight=1)

        # 第 0 行：扩展名过滤复选框
        chk_ply = ttk.Checkbutton(
            frame_bottom,
            text=".ply",
            variable=self.show_ply_var,
            command=self.scan_models,
        )
        chk_ply.grid(row=0, column=0, sticky="w", padx=(0, 8))

        chk_obj = ttk.Checkbutton(
            frame_bottom,
            text=".obj",
            variable=self.show_obj_var,
            command=self.scan_models,
        )
        chk_obj.grid(row=0, column=0, sticky="w", padx=(60, 8))

        chk_npz = ttk.Checkbutton(
            frame_bottom,
            text=".npz",
            variable=self.show_npz_var,
            command=self.scan_models,
        )
        chk_npz.grid(row=0, column=0, sticky="w", padx=(120, 8))

        # 第 1 行：名称前缀过滤复选框
        chk_splat = ttk.Checkbutton(
            frame_bottom,
            text="splat*",
            variable=self.filter_splat_var,
            command=self.scan_models,
        )
        chk_splat.grid(row=1, column=0, sticky="w", padx=(0, 8))

        chk_mesh = ttk.Checkbutton(
            frame_bottom,
            text="mesh*",
            variable=self.filter_mesh_var,
            command=self.scan_models,
        )
        chk_mesh.grid(row=1, column=0, sticky="w", padx=(80, 8))

        # 第 2 行：.npz 打开方式（静态 / 动态），互斥
        def on_npz_method_changed(changed_var):
            """
            保证 method_npz_final_var 和 method_npz_online_var 互斥且至少选一个。
            changed_var: 被用户点击的 BooleanVar。
            """
            # 若用户把某个选项从 False 点成 True，则将另一个设为 False
            if changed_var.get():
                if changed_var is self.method_npz_final_var:
                    self.method_npz_online_var.set(False)
                else:
                    self.method_npz_final_var.set(False)
            else:
                # 用户尝试取消当前选项，如果另一个也是 False，则强制保持当前为 True，保证至少一个
                other = self.method_npz_online_var if changed_var is self.method_npz_final_var else self.method_npz_final_var
                if not other.get():
                    changed_var.set(True)

        chk_npz_final = ttk.Checkbutton(
            frame_bottom,
            text="npz: 静态 (final)",
            variable=self.method_npz_final_var,
            command=lambda: on_npz_method_changed(self.method_npz_final_var),
        )
        chk_npz_final.grid(row=2, column=0, sticky="w", padx=(0, 8))

        chk_npz_online = ttk.Checkbutton(
            frame_bottom,
            text="npz: 动态 (online)",
            variable=self.method_npz_online_var,
            command=lambda: on_npz_method_changed(self.method_npz_online_var),
        )
        chk_npz_online.grid(row=2, column=0, sticky="w", padx=(130, 8))

        # 第 3 行：刷新按钮 + 状态栏
        self.btn_refresh = ttk.Button(frame_bottom, text="刷新 (Rescan)", command=self.scan_models)
        self.btn_refresh.grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(4, 0))

        self.status_var = tk.StringVar(value="")
        self.lbl_status = ttk.Label(frame_bottom, textvariable=self.status_var, anchor="w")
        self.lbl_status.grid(row=3, column=1, sticky="ew", padx=(12, 0), pady=(4, 0))

    # ---------------- 目录扫描 ----------------
    def scan_models(self):
        """扫描 experiments 目录，查找 .ply / .obj / .npz 文件。"""
        self.tree.delete(*self.tree.get_children())

        if not os.path.isdir(self.experiments_dir):
            msg = f"experiments 目录不存在：{self.experiments_dir}"
            self.status_var.set(msg)
            messagebox.showwarning("目录不存在", msg)
            return

        # 根据复选框决定要显示的扩展名
        exts = set()
        if self.show_ply_var.get():
            exts.add(".ply")
        if self.show_obj_var.get():
            exts.add(".obj")
        if self.show_npz_var.get():
            exts.add(".npz")

        # 名称前缀过滤：若至少勾选一个，则只保留以这些前缀开头的文件；否则不过滤前缀
        prefix_filters = []
        if self.filter_splat_var.get():
            prefix_filters.append("splat")
        if self.filter_mesh_var.get():
            prefix_filters.append("mesh")

        # 按目录收集文件，便于在每个子目录内按修改时间降序排序
        files_by_dir = {}

        for root_dir, _, files in os.walk(self.experiments_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if exts and ext not in exts:
                    continue

                # 前缀过滤：只作用于 .ply / .obj，.npz 不受文件名前缀约束
                if ext in (".ply", ".obj") and prefix_filters:
                    lower_name = name.lower()
                    if not any(lower_name.startswith(p) for p in prefix_filters):
                        continue

                full_path = os.path.join(root_dir, name)
                try:
                    size_bytes = os.path.getsize(full_path)
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    size_bytes = 0
                    mtime = 0.0

                rel_path = os.path.relpath(full_path, self.experiments_dir)
                rel_dir = os.path.dirname(rel_path)
                file_name = os.path.basename(rel_path)
                files_by_dir.setdefault(rel_dir, []).append(
                    (file_name, full_path, size_bytes, mtime)
                )

        # 逐目录创建节点，并在每个目录内按 mtime 降序插入文件
        for rel_dir in sorted(files_by_dir.keys()):
            # 创建目录节点链
            parent = ""
            if rel_dir not in ("", "."):
                parts = rel_dir.split(os.sep)
                current_parts = []
                for folder in parts:
                    current_parts.append(folder)
                    node_id = "/".join(current_parts)
                    if not self.tree.exists(node_id):
                        self.tree.insert(
                            parent,
                            "end",
                            iid=node_id,
                            text=folder,
                            values=("", "", ""),
                            open=False,
                        )
                    parent = node_id

            # 在该目录下按修改时间排序后插入文件（最新在前）
            files_list = files_by_dir[rel_dir]
            files_list.sort(key=lambda x: x[3], reverse=True)
            for file_name, full_path, size_bytes, mtime in files_list:
                size_mb = size_bytes / (1024 * 1024.0)
                size_str = f"{size_mb:.2f}"
                mtime_str = (
                    datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    if mtime > 0
                    else ""
                )
                self.tree.insert(
                    parent,
                    "end",
                    text=file_name,
                    values=(size_str, mtime_str, full_path),
                )

        # 递归展开所有节点，默认全展开
        def expand_all(parent=""):
            children = self.tree.get_children(parent)
            for child in children:
                self.tree.item(child, open=True)
                expand_all(child)

        expand_all("")

        # 更新状态栏
        exts_desc = []
        if ".ply" in exts:
            exts_desc.append(".ply")
        if ".obj" in exts:
            exts_desc.append(".obj")
        if ".npz" in exts:
            exts_desc.append(".npz")
        if not exts_desc:
            exts_text = "无扩展名被选中"
        else:
            exts_text = ", ".join(exts_desc)

        self.status_var.set(f"已扫描：{self.experiments_dir}  |  过滤：{exts_text}")

    # ---------------- 事件处理 ----------------
    def on_tree_double_click(self, event):
        """双击某一条目时，打开对应的模型查看器进程。"""
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return

        values = self.tree.item(item_id, "values")
        if not values or len(values) < 3:
            # 可能是纯目录节点或数据异常
            # 对目录节点，切换展开/折叠状态
            item = self.tree.item(item_id)
            has_children = bool(self.tree.get_children(item_id))
            if has_children:
                self.tree.item(item_id, open=not item.get("open", False))
            return

        size_str, mtime_str, full_path = values[0], values[1], values[2]
        if not full_path or os.path.isdir(full_path):
            # 目录节点：仅展开/折叠
            item = self.tree.item(item_id)
            self.tree.item(item_id, open=not item.get("open", False))
            return

        self.open_viewer_process(full_path)

    def open_viewer_process(self, full_path):
        """启动独立进程，显示指定模型或 .npz 结果。"""
        if not os.path.isfile(full_path):
            messagebox.showerror("文件不存在", f"无法找到文件：\n{full_path}")
            return

        _, ext = os.path.splitext(full_path)
        ext = ext.lower()

        try:
            if ext in [".ply", ".obj"]:
                # 传统 mesh：用 Trimesh 查看
                p = mp.Process(target=view_model, args=(full_path,))
                p.daemon = False  # 允许独立存在，直到窗口关闭
                p.start()
                self.viewer_processes.append(p)
                self.status_var.set(f"打开模型：{os.path.relpath(full_path, self.experiments_dir)}")
            elif ext == ".npz":
                # .npz：根据当前选择，调用 final_recon.py 或 online_recon.py
                self.open_npz_viewer(full_path)
            else:
                messagebox.showwarning("不支持的文件类型", f"暂不支持打开该类型文件：\n{full_path}")
        except Exception as e:
            messagebox.showerror("启动查看器失败", f"无法启动查看器进程：\n{e}")

    def open_npz_viewer(self, full_path):
        """
        根据 .npz 打开方式选择，调用 viz_scripts/final_recon.py 或
        viz_scripts/online_recon.py 在独立进程中进行可视化。
        """
        # 保证互斥和至少一个选中（防御性检查，正常情况下 UI 已经保证）
        if not self.method_npz_final_var.get() and not self.method_npz_online_var.get():
            # 默认回退到静态查看
            self.method_npz_final_var.set(True)
        if self.method_npz_final_var.get() and self.method_npz_online_var.get():
            # 若意外同时为 True，则优先静态，关闭动态
            self.method_npz_online_var.set(False)

        # 构造脚本路径
        final_script = os.path.join(self.base_dir, "viz_scripts", "final_recon.py")
        online_script = os.path.join(self.base_dir, "viz_scripts", "online_recon.py")

        if self.method_npz_final_var.get():
            script_path = final_script
            mode_desc = "静态 (final_recon)"
        else:
            script_path = online_script
            mode_desc = "动态 (online_recon)"

        if not os.path.isfile(script_path):
            messagebox.showerror("脚本不存在", f"找不到可视化脚本：\n{script_path}")
            return

        try:
            # 使用 subprocess.Popen 启动独立 Python 进程，可避免多进程 pickling 问题
            p = subprocess.Popen(
                [sys.executable, script_path, full_path],
                cwd=self.base_dir,
            )
            self.viewer_processes.append(p)
            rel = os.path.relpath(full_path, self.experiments_dir)
            self.status_var.set(f"打开 .npz（{mode_desc}）：{rel}")
        except Exception as e:
            messagebox.showerror("启动 .npz 查看器失败", f"无法启动 .npz 查看器进程：\n{e}")


def main():
    # 在 Linux 上默认使用 fork，一般没问题；若需兼容性更好可选择 spawn
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        # start method 已经设置过，忽略
        pass

    root = tk.Tk()
    app = ModelBrowserApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


