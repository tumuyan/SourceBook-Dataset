import os
import sys
from PIL import Image, UnidentifiedImageError
import imagehash
import argparse
from tqdm import tqdm # 导入 tqdm
import time # 导入 time 模块用于延时
import concurrent.futures # 导入 concurrent.futures 模块
import threading # 导入 threading 模块用于锁

# 支持的图片文件扩展名
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

# 使用 threading.Lock 保护共享资源 (canonical_hashes_map)
canonical_map_lock = threading.Lock()
# Stores representative hashes of images we keep.
# key: Representative hash object (obtained by sorting hash strings)
# value: The *relative* path of the first file found for this content.
# This is shared state accessed by workers (indirectly via main thread processing results)
canonical_hashes_map = {}


def is_image_file(filepath):
    """检查文件是否是支持的图片文件"""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in SUPPORTED_EXTENSIONS

def calculate_all_8_hashes(image):
    """
    计算图片在所有8种标准方向下的dHash值。
    返回一个包含最多8个独特imagehash对象的集合，或在发生错误时返回None。
    """
    hashes = set()
    img_transformed = None # Variable to hold intermediate images for closing

    try:
        # Ensure image is RGB or L for hashing, convert if necessary
        if image.mode not in ('RGB', 'L'):
             image = image.convert('RGB')

        # Transformations to compute hashes for the 8 orientations
        transformations = [
            image, # Original
            image.rotate(90, expand=True),
            image.rotate(180, expand=True),
            image.rotate(270, expand=True),
            image.transpose(Image.FLIP_LEFT_RIGHT),
            image.transpose(Image.FLIP_TOP_BOTTOM),
            image.transpose(Image.TRANSPOSE), # R90 + H Flip
            image.transpose(Image.TRANSVERSE), # R270 + H Flip
        ]

        for img_t in transformations:
             try:
                hashes.add(imagehash.dhash(img_t))
             except Exception as hash_e:
                 # If any single transformation fails to hash, skip it but continue with others
                 # This specific hash failure isn't a total error for the file unless no hashes are generated
                 pass

    except Exception as e:
        # Handle errors during the overall process (e.g., initial image mode conversion failure, or major PIL transform error)
        # print(f"\n警告: 无法为图片计算全部8种方向的哈希值 (可能格式问题或损坏): {e}", file=sys.stderr) # Error logging handled by caller
        return None # Indicate total failure for this file


    # Return the set of unique hash values. If set is empty, treat as failure.
    if not hashes:
         return None # Indicate failure


    return hashes


# Function to process a single image file - will be run by worker threads
def process_single_image(filepath, root_dir, min_resolution):
    """
    处理单个图片文件：检查分辨率、计算哈希。
    不执行删除操作，不修改全局状态。
    返回文件的处理结果信息。
    """
    try:
        # Calculate relative path for reporting
        try:
             relative_filepath = os.path.relpath(filepath, root_dir)
        except ValueError:
             relative_filepath = filepath # Fallback

        img = Image.open(filepath)
        img_width, img_height = img.size

        # Check resolution (still done by worker for efficiency)
        if img_width < min_resolution[0] or img_height < min_resolution[1]:
            img.close()
            return {'status': 'low_res', 'filepath': filepath, 'relative_filepath': relative_filepath, 'details': (img_width, img_height)}

        # If resolution is ok, calculate hashes
        all_hashes = calculate_all_8_hashes(img)
        img.close()

        if all_hashes is None or not all_hashes:
            # Hashing failed or returned empty set
            return {'status': 'error_hash', 'filepath': filepath, 'relative_filepath': relative_filepath, 'details': 'Hashing failed'}

        # Successfully calculated hashes, return them for duplicate check by main thread
        return {'status': 'hash_success', 'filepath': filepath, 'relative_filepath': relative_filepath, 'details': all_hashes}

    except FileNotFoundError:
         return {'status': 'error_not_found', 'filepath': filepath, 'relative_filepath': relative_filepath, 'details': 'File not found or deleted'}
    except UnidentifiedImageError:
         return {'status': 'error_bad_format', 'filepath': filepath, 'relative_filepath': relative_filepath, 'details': 'Invalid image format or corrupted'}
    except Exception as e:
         # Catch any other unexpected errors during processing
         return {'status': 'error_other', 'filepath': filepath, 'relative_filepath': relative_filepath, 'details': str(e)}


def clean_images(root_dir, min_resolution, log_level=0, try_run=False, delay_seconds=0.0, max_depth=-1, num_workers=None):
    """
    清理指定目录及其子目录下的图片文件。

    清理规则：
    1. 删除分辨率低于指定数值的图片。
    2. 删除内容与其他图片重复（需要考虑翻转）的图片。

    Args:
        root_dir (str): 开始清理的根目录路径。
        min_resolution (tuple): 最小分辨率，格式为 (宽度, 高度)。
        log_level (int): 打印信息等级 (0: 全部文件处理信息+进度条, 1: 简洁文件处理信息+进度条, 2: 只打印进度条)。
        try_run (bool): 如果为True，则只模拟清理过程，不实际删除文件。
        delay_seconds (float): 处理每个文件后模拟延时的秒数。
        max_depth (int): 限制扫描子目录的层级深度 (0: 仅根目录, 1: 根目录及其一级子目录, -1: 不限制)。
        num_workers (int, optional): 并行处理的线程数。默认 None，使用 os.cpu_count()。
    """
    if not os.path.isdir(root_dir):
        print(f"错误：目录不存在或不是一个有效目录: {root_dir}", file=sys.stderr)
        return

    # Initial messages
    print(f"开始扫描和清理目录: {root_dir}")
    print(f"最小分辨率要求: {min_resolution[0]}x{min_resolution[1]}")
    if max_depth != -1:
         print(f"最大扫描深度: {max_depth} 层 (0: 仅根目录)")
    else:
         print("最大扫描深度: 不限制")
    if try_run:
        print("--- TRY RUN 模式: 文件不会被实际删除 ---")
    if delay_seconds > 0:
         print(f"--- 每个文件处理后模拟延时: {delay_seconds:.2f} 秒 ---")
    if num_workers is not None:
         print(f"--- 使用 {num_workers} 个工作线程 ---")
    else:
         print(f"--- 使用默认数量 ({os.cpu_count() if os.cpu_count() is not None else '?'}) 个工作线程 ---")
    print("-" * 30)

    # Pass 1: Collect all image files to get the total count for tqdm
    if log_level <= 1:
       print("扫描目录以收集图片文件列表...")

    image_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Calculate current depth relative to root_dir
        try:
             relative_dir = os.path.relpath(dirpath, root_dir)
             if relative_dir == '.':
                 current_depth = 0
             else:
                 current_depth = relative_dir.count(os.sep) + 1

        except ValueError:
            if log_level == 0:
                print(f"警告: 无法计算目录 '{dirpath}' 相对于根目录 '{root_dir}' 的深度 (不同驱动器?), 跳过该目录。", file=sys.stderr)
            del dirnames[:]
            continue
        except Exception as e:
             if log_level == 0:
                 print(f"警告: 计算目录 '{dirpath}' 深度时发生未知错误: {e}", file=sys.stderr)
             del dirnames[:]
             continue

        # Check depth limit
        if max_depth != -1 and current_depth > max_depth:
            if log_level == 0:
                print(f"跳过目录 '{dirpath}' (深度 {current_depth} > 最大深度 {max_depth})", file=sys.stderr)
            del dirnames[:]
            continue

        # Collect image files in this directory
        for filename in filenames:
             filepath = os.path.join(dirpath, filename)
             if is_image_file(filepath):
                 image_files.append(filepath)

    total_potential_images = len(image_files) # Total files found after filtering

    if log_level <= 1:
         if total_potential_images > 0:
            print(f"扫描完成，找到 {total_potential_images} 个符合条件的图片文件进行处理。")
         else:
            print("扫描完成，未找到符合条件的图片文件进行处理，或者所有符合条件的目录都超过了最大深度。")


    # Initialize counters for Pass 2 (Processing phase)
    total_images_processed_attempted = 0
    identified_low_res = 0
    identified_duplicates = 0
    identified_errors = 0
    # The global canonical_hashes_map is used here


    # Pass 2: Process files using ThreadPoolExecutor and tqdm
    tqdm_desc_base = "处理进度"
    if try_run:
         tqdm_desc_base = "检查进度 (Try Run)"

    # Use ThreadPoolExecutor
    # max_workers=None uses os.cpu_count()
    # thread_name_prefix can help debugging
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="ImgWorker") as executor:
        # Submit all tasks and store the futures
        futures = {executor.submit(process_single_image, filepath, root_dir, min_resolution): filepath for filepath in image_files}

        # Process results as they complete using tqdm
        # Disable tqdm bar if there are no files to process
        with tqdm(concurrent.futures.as_completed(futures), total=total_potential_images, desc=tqdm_desc_base, unit="file", file=sys.stdout, disable=(total_potential_images == 0)) as t:
            for future in t:
                # Ensure total_images_processed_attempted matches total tasks submitted
                total_images_processed_attempted += 1

                filepath = futures[future] # Get the original filepath associated with the future
                try:
                    # Get result from the worker
                    result = future.result() # This will raise exceptions if worker did

                    # Process the result returned by the worker
                    action_taken = None # 'identified_low_res', 'identified_duplicate', 'kept', 'error_...'
                    message_to_print = None

                    if result['status'] == 'low_res':
                        action_taken = 'identified_low_res'
                        identified_low_res += 1
                        delete_msg_prefix = "标记删除" if try_run else "删除"
                        img_width, img_height = result['details']
                        if log_level == 0:
                            message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: {delete_msg_prefix} (分辨率过低 {img_width}x{img_height})"
                        else: # log_level 1
                             message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: {delete_msg_prefix} (分辨率过低)"

                        # Perform actual deletion if not try_run
                        if not try_run:
                            try:
                                os.remove(filepath)
                            except OSError as e:
                                message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: [!] 错误 (删除失败: {e})"
                                action_taken = 'error_delete'
                                identified_errors += 1 # Count deletion failure as an error

                    elif result['status'] == 'hash_success':
                        all_hashes = result['details']
                        # Find the representative hash object from the set
                        # This must be done by the main thread *after* getting hashes from worker
                        # Convert hashes to strings, find the minimum string, then get the corresponding hash object
                        # This handles the TypeError when sorting ImageHash objects directly
                        try:
                            hash_string_pairs = [(str(h), h) for h in all_hashes]
                            representative_hash_object = min(hash_string_pairs, key=lambda item: item[0])[1]
                        except Exception as e:
                            action_taken = 'error_represent_hash'
                            message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: [!] 错误 (确定代表哈希异常: {e})"
                            identified_errors += 1
                            representative_hash_object = None # Indicate failure

                        if representative_hash_object is not None:
                             # --- Duplicate Check (Requires Lock) ---
                             is_duplicate = False
                             matching_canonical_relative_path = None

                             # Acquire lock before accessing the shared map
                             with canonical_map_lock:
                                if representative_hash_object in canonical_hashes_map:
                                     is_duplicate = True
                                     matching_canonical_relative_path = canonical_hashes_map[representative_hash_object]
                                else:
                                     # Not a duplicate, add to map using the representative hash object
                                     canonical_hashes_map[representative_hash_object] = result['relative_filepath']

                             # Release lock automatically by 'with' statement

                             if is_duplicate:
                                 action_taken = 'identified_duplicate'
                                 identified_duplicates += 1
                                 delete_msg_prefix = "标记删除" if try_run else "删除"
                                 if log_level == 0:
                                     message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: {delete_msg_prefix} (重复, 内容与 {matching_canonical_relative_path} 相同)"
                                 else: # log_level 1
                                     message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: {delete_msg_prefix} (重复)"

                                 # Perform actual deletion if not try_run
                                 if not try_run:
                                     try:
                                         os.remove(filepath)
                                     except OSError as e:
                                         message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: [!] 错误 (删除失败: {e})"
                                         action_taken = 'error_delete' # Overwrite action_taken
                                         identified_errors += 1 # Count deletion failure as an error

                             else: # Not a duplicate (based on 8-way check and map check)
                                 action_taken = 'kept'
                                 if log_level == 0:
                                     message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: [+] 保留 (新内容)"

                         # If representative_hash_object was None due to error, message was already set above


                    elif result['status'].startswith('error_'):
                         # Worker reported an error (not low res, not hash_success)
                         action_taken = result['status'] # e.g., 'error_not_found', 'error_bad_format', 'error_other'
                         message_to_print = f"({total_images_processed_attempted}/{total_potential_images}) {result['relative_filepath']}: [!] 错误 ({result['details']})"
                         identified_errors += 1


                    # Print message above the tqdm bar based on log_level and action
                    if message_to_print:
                         if log_level == 0:
                              t.write(message_to_print)
                         elif log_level == 1:
                              # Only print if it was an identified deletion or an error
                              error_action_types = ('error_hash', 'error_not_found', 'error_bad_format',
                                                     'error_other', 'error_delete', 'error_represent_hash') # Include represent_hash error
                              if action_taken in ('identified_low_res', 'identified_duplicate') or action_taken in error_action_types:
                                   t.write(message_to_print)
                         # log_level 2 prints nothing via t.write()


                except Exception as exc:
                    # Catch any exception raised by the worker thread *after* it returned a future
                    # This is less common if worker handles its own exceptions and returns a status
                    # But safety first.
                    identified_errors += 1
                    relative_filepath_for_error = os.path.relpath(filepath, root_dir) # Try to get relative path for error msg
                    t.write(f"({total_images_processed_attempted}/{total_potential_images}) {relative_filepath_for_error}: [!] 错误 (工作线程异常: {exc})", file=sys.stderr)

                # Apply delay *after* processing the result from the worker
                # This delay makes the tqdm updates visible for observation, not for simulating work time
                if delay_seconds > 0:
                     time.sleep(delay_seconds)

            # End of for future in t loop
        # End of with tqdm as t block - tqdm bar closes automatically

    # End of with ThreadPoolExecutor block - workers shut down


    # Print final report after all tasks are completed and results processed
    total_identified_for_deletion = identified_low_res + identified_duplicates
    # Remaining images are those attempted minus those identified for deletion and those with errors
    remaining_images = total_images_processed_attempted - total_identified_for_deletion - identified_errors

    # Calculate percentages based on total_images_processed_attempted
    if total_images_processed_attempted > 0:
        low_res_pct = (identified_low_res / total_images_processed_attempted) * 100
        duplicates_pct = (identified_duplicates / total_images_processed_attempted) * 100
        to_delete_pct = (total_identified_for_deletion / total_images_processed_attempted) * 100
        errors_pct = (identified_errors / total_images_processed_attempted) * 100
        remaining_pct = (remaining_images / total_images_processed_attempted) * 100
    else:
        # Handle case where no files were processed
        low_res_pct = 0.0
        duplicates_pct = 0.0
        to_delete_pct = 0.0
        errors_pct = 0.0
        remaining_pct = 0.0


    print("\n--- 清理完成统计 ---")
    if try_run:
        print("(Try Run 模式: 以下数字表示将被处理的文件)")
    print(f"扫描到的符合条件的图片文件总数 (受深度限制): {total_potential_images}")
    print(f"尝试处理的图片文件总数: {total_images_processed_attempted}")
    print(f"识别出分辨率过低图片数: {identified_low_res} ({low_res_pct:.2f}%)")
    print(f"识别出重复图片数 (含8种方向): {identified_duplicates} ({duplicates_pct:.2f}%)")
    print(f"总计识别出待处理图片数: {total_identified_for_deletion} ({to_delete_pct:.2f}%)")
    print(f"处理过程中发生错误文件数: {identified_errors} ({errors_pct:.2f}%)")
    print(f"预计保留图片总数: {remaining_images} ({remaining_pct:.2f}%)")
    print("----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理目录下的图片文件，删除低分辨率和重复图片（含8种方向翻转/旋转）。")
    parser.add_argument("root_dir", help="要清理的根目录路径。")
    parser.add_argument("-r", "--resolution", required=True,
                        help="最小分辨率，格式为 宽度x高度 (例如: 800x600)。")
    parser.add_argument("-l", "--log-level", type=int, default=0, choices=[0, 1, 2],
                        help="打印信息等级 (0: 全部文件处理信息+进度条, 1: 简洁文件处理信息+进度条, 2: 只打印进度条)。默认: 0")
    parser.add_argument("-t", "--try-run", action="store_true",
                        help="只检查文件并报告将被处理的文件，不实际删除。")
    parser.add_argument("-d", "--delay", type=float, default=0.0,
                        help="处理每个文件后模拟延时的秒数 (浮点数, 例如 0.1)。用于观察进度条。默认: 0.0")
    parser.add_argument("-depth", "--max-depth", type=int, default=-1,
                        help="限制扫描子目录的层级深度 (0: 仅根目录, 1: 根目录及其一级子目录, -1: 不限制)。默认: -1")
    # Add num_workers parameter
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="并行处理的线程数。默认使用系统中CPU核心数。")


    args = parser.parse_args()

    # Parse resolution argument
    try:
        width, height = map(int, args.resolution.split('x'))
        if width <= 0 or height <= 0:
             raise ValueError("宽度和高度必须大于0")
        min_res = (width, height)
    except ValueError as e:
        print(f"错误: 无效的分辨率格式 '{args.resolution}'. 请使用 宽度x高度 的格式，例如 800x600。", file=sys.stderr)
        print(f"详细信息: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate max_depth
    if args.max_depth < -1:
         print(f"错误: 无效的最大深度值 '{args.max_depth}'. 必须是 -1 或大于等于 0 的整数。", file=sys.stderr)
         sys.exit(1)

    # Validate delay
    if args.delay < 0:
         print(f"错误: 无效的延时值 '{args.delay}'. 必须大于等于 0。", file=sys.stderr)
         sys.exit(1)

    # Validate workers
    if args.workers is not None and args.workers <= 0:
         print(f"错误: 无效的工作线程数 '{args.workers}'. 必须是正整数。", file=sys.stderr)
         sys.exit(1)


    # Normalize root_dir path for consistent relative path calculation
    args.root_dir = os.path.abspath(args.root_dir)


    # Execute cleaning
    try:
        clean_images(args.root_dir, min_res,
                     log_level=args.log_level, try_run=args.try_run,
                     delay_seconds=args.delay, max_depth=args.max_depth,
                     num_workers=args.workers) # Pass num_workers
    except KeyboardInterrupt:
        print("\n清理过程被用户中断。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch general exceptions not handled inside the loop
        print(f"\n处理过程中发生未捕获的错误: {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc() # Optional: print full traceback for debugging
        sys.exit(1)