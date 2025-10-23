import os
import sys
from PIL import Image, UnidentifiedImageError
import imagehash
import argparse
from tqdm import tqdm # 导入 tqdm
import time # 导入 time 模块用于延时

# 支持的图片文件扩展名
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

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
             # Converting to RGB is a robust approach for dhash.
             image = image.convert('RGB')

        # List of transformations to apply (using PIL transpose/rotate constants)
        # Tuples are (transform_type, argument, expand_boolean)
        # Note: Some transformations are compositions (e.g., Vertical Flip is R180 + H Flip)
        # We list out the 8 unique orientations as direct or composed transforms
        # Orientations:
        # 1. 0 degrees
        # 2. 90 degrees
        # 3. 180 degrees
        # 4. 270 degrees
        # 5. Horizontal Flip (along Y axis)
        # 6. Vertical Flip (along X axis) - same as R180 + H Flip
        # 7. Transpose (Top-Left to Bottom-Right) - same as R90 + H Flip
        # 8. Transverse (Top-Right to Bottom-Left) - same as R270 + H Flip
        # PIL's FLIP_TOP_BOTTOM is Vertical Flip
        # PIL's TRANSPOSE is R90 + H Flip
        # PIL's TRANSVERSE is R270 + H Flip

        transformations = [
            image, # Original
            image.rotate(90, expand=True),
            image.rotate(180, expand=True), # expand=True is safe even if not needed
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
                 # Handle hashing errors for a specific transformation
                 # print(f"Debug: Hashing failed for a transformation: {hash_e}", file=sys.stderr) # Optional debug
                 pass # Skip this specific hash calculation but continue

    except Exception as e:
        # Handle errors during the overall process (e.g., initial image mode conversion failure, or PIL transform error)
        # Use stderr to not interfere with tqdm output
        print(f"\n警告: 无法为图片计算全部8种方向的哈希值 (可能格式问题或损坏): {e}", file=sys.stderr)
        return None # Indicate total failure for this file


    # Ensure intermediate images are closed
    # Note: The original 'image' should not be closed here if it was passed from outside
    # We need to close only the images created within this function
    # A more robust way is to create copies or manage resources carefully.
    # For now, assuming the original image is managed by the caller, and closing intermediates.
    # The list comprehension approach above with temporary variables might be cleaner for closing.
    # Let's revert to creating and closing individually within the loop for clarity and resource management.

    hashes = set() # Reset hashes set
    try:
        # Ensure image is RGB or L for hashing, convert if necessary
        if image.mode not in ('RGB', 'L'):
             image = image.convert('RGB') # This might return a new image or modify in-place (depends on PIL)

        # 1. Original (0 degree)
        hashes.add(imagehash.dhash(image))

        # 2. Rotated 90 degrees
        img_transformed = image.rotate(90, expand=True)
        hashes.add(imagehash.dhash(img_transformed))
        img_transformed.close()

        # 3. Rotated 180 degrees
        img_transformed = image.rotate(180, expand=True)
        hashes.add(imagehash.dhash(img_transformed))
        img_transformed.close()

        # 4. Rotated 270 degrees
        img_transformed = image.rotate(270, expand=True)
        hashes.add(imagehash.dhash(img_transformed))
        img_transformed.close()

        # 5. Horizontal Flip
        img_transformed = image.transpose(Image.FLIP_LEFT_RIGHT)
        hashes.add(imagehash.dhash(img_transformed))
        img_transformed.close()

        # 6. Vertical Flip
        img_transformed = image.transpose(Image.FLIP_TOP_BOTTOM)
        hashes.add(imagehash.dhash(img_transformed))
        img_transformed.close()

        # 7. Transpose (R90 + H Flip)
        img_transformed = image.transpose(Image.TRANSPOSE)
        hashes.add(imagehash.dhash(img_transformed))
        img_transformed.close()

        # 8. Transverse (R270 + H Flip)
        img_transformed = image.transpose(Image.TRANSVERSE)
        hashes.add(imagehash.dhash(img_transformed))
        img_transformed.close()


    except Exception as e:
        # Handle errors during the overall process (e.g., initial image mode conversion failure, or PIL transform error)
        # Use stderr to not interfere with tqdm output
        print(f"\n警告: 无法为图片计算全部8种方向的哈希值 (可能格式问题或损坏): {e}", file=sys.stderr)
        return None # Indicate total failure for this file

    # Return the set of unique hash values
    if not hashes: # If set is empty despite no top-level error
         print(f"\n警告: 成功打开图片但未能计算任何有效哈希。", file=sys.stderr)
         return None # Treat as failure


    return hashes


# Keep the old calculate_hashes function name for clarity, but it now does the 8-way logic
calculate_hashes = calculate_all_8_hashes


def clean_images(root_dir, min_resolution, log_level=0, try_run=False, delay_seconds=0.0, max_depth=-1):
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
    """
    if not os.path.isdir(root_dir):
        print(f"错误：目录不存在或不是一个有效目录: {root_dir}", file=sys.stderr)
        return

    # Initial messages before tqdm starts
    print(f"开始扫描和清理目录: {root_dir}")
    print(f"最小分辨率要求: {min_resolution[0]}x{min_resolution[1]}")
    if max_depth != -1:
         print(f"最大扫描深度: {max_depth} 层 (0: 仅根目录)")
    else:
         print("最大扫描深度: 不限制")
    if try_run:
        print("--- TRY RUN 模式: 文件不会被实际删除 ---")
    if delay_seconds > 0:
         print(f"--- 每个文件处理后延时: {delay_seconds:.2f} 秒 ---")
    print("-" * 30)

    # Pass 1: Collect all image files to get the total count for tqdm
    # This message is only needed if log_level <= 1 (to show some activity before the bar)
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
            # Handle different drives on Windows
            # Only print this warning for log_level 0
            if log_level == 0:
                print(f"警告: 无法计算目录 '{dirpath}' 相对于根目录 '{root_dir}' 的深度 (不同驱动器?), 跳过该目录。", file=sys.stderr)
            del dirnames[:] # Prevent os.walk from descending further
            continue
        except Exception as e:
             # Only print this warning for log_level 0
             if log_level == 0:
                 print(f"警告: 计算目录 '{dirpath}' 深度时发生未知错误: {e}", file=sys.stderr)
             del dirnames[:] # Prevent os.walk from descending further
             continue


        # Check if current directory depth exceeds max_depth (unless max_depth is -1)
        # Apply check *before* processing files in this directory
        if max_depth != -1 and current_depth > max_depth:
            # Only print skip message for log_level 0
            if log_level == 0:
                print(f"跳过目录 '{dirpath}' (深度 {current_depth} > 最大深度 {max_depth})", file=sys.stderr)

            del dirnames[:] # Prevent os.walk from descending further into this directory's subdirs
            continue # Skip processing files in *this* directory

        # If not exceeding depth, collect image files in this directory
        for filename in filenames:
             filepath = os.path.join(dirpath, filename)
             if is_image_file(filepath):
                 image_files.append(filepath)

    total_potential_images = len(image_files) # Total files found after filtering by depth and extension

    # Print summary of scanning results if log_level <= 1
    # This message is helpful even if total_potential_images is 0, just adjust text
    if log_level <= 1:
         if total_potential_images > 0:
            print(f"扫描完成，找到 {total_potential_images} 个符合条件的图片文件进行处理。")
         else:
            print("扫描完成，未找到符合条件的图片文件进行处理，或者所有符合条件的目录都超过了最大深度。")


    # Initialize counters for Pass 2
    # total_images_processed_attempted: number of files from image_files list we try to open
    total_images_processed_attempted = 0
    identified_low_res = 0
    identified_duplicates = 0
    identified_errors = 0 # Counter for files that caused processing errors

    # Stores representative hashes of images we keep.
    # key: Representative hash object (obtained by sorting hash strings)
    # value: The *relative* path of the first file found for this content.
    canonical_hashes_map = {}


    # Pass 2: Process files with tqdm
    tqdm_desc_base = "处理进度"
    if try_run:
         tqdm_desc_base = "检查进度 (Try Run)"

    # Decide if we show the tqdm bar. Show if there are files to process.
    # The bar should be shown for log_level 0, 1, 2 if there are files.
    # Disable the bar ONLY if there are NO files to process.
    with tqdm(enumerate(image_files), total=total_potential_images, desc=tqdm_desc_base, unit="file", file=sys.stdout, disable=(total_potential_images == 0)) as t:
        for processed_count, filepath in t: # tqdm provides 0-indexed index

            # Calculate relative path for messages
            try:
                 relative_filepath = os.path.relpath(filepath, root_dir)
            except ValueError:
                 # Should not happen if root_dir and filepath are from the same os.walk, but safety first
                 relative_filepath = filepath # Fallback to full path if relative calculation fails


            # We increment this because we are now processing a file from the collected list
            total_images_processed_attempted += 1

            action_taken = None # 'identified_low_res', 'identified_duplicate', 'kept', 'error_...'
            message_to_print = None # Message content for tqdm.write()

            try:
                # Open image
                # Note: If a file was deleted between Pass 1 and Pass 2, FileNotFoundError occurs here.
                img = Image.open(filepath)
                img_width, img_height = img.size

                # Check resolution
                if img_width < min_resolution[0] or img_height < min_resolution[1]:
                    action_taken = 'identified_low_res'
                    identified_low_res += 1
                    delete_msg_prefix = "标记删除" if try_run else "删除"
                    # Log_level 0 includes dimensions, Log_level 1 is concise
                    if log_level == 0:
                        message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: {delete_msg_prefix} (分辨率过低 {img_width}x{img_height})"
                    else: # log_level 1
                         message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: {delete_msg_prefix} (分辨率过低)"


                    img.close() # Ensure close
                    if not try_run:
                        try:
                            os.remove(filepath)
                        except OSError as e:
                             # Error specifically during deletion
                            message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [!] 错误 (删除失败: {e})"
                            action_taken = 'error_delete'
                            identified_errors += 1

                else: # Resolution OK, check duplicates using 8-way hashing
                    # Calculate all 8 orientation hashes
                    all_hashes = calculate_hashes(img) # This now calls calculate_all_8_hashes
                    img.close() # Ensure close

                    if all_hashes is None: # Hashing failed entirely for this file (error already printed in calculate_hashes)
                         action_taken = 'error_hash_failed'
                         # message_to_print is set below based on action_taken
                         identified_errors += 1
                    elif not all_hashes: # Empty set returned - implies no valid hashes could be calculated
                         action_taken = 'error_hash_partial'
                         # message_to_print is set below
                         identified_errors += 1 # Count as error since we can't check duplicates
                    else:
                        # Find the representative hash object from the set
                        # Convert hashes to strings, find the minimum string, then get the corresponding hash object
                        # This handles the TypeError when sorting ImageHash objects directly
                        try:
                            hash_string_pairs = [(str(h), h) for h in all_hashes]
                            # Sort by the string representation
                            representative_hash_object = min(hash_string_pairs, key=lambda item: item[0])[1]
                        except Exception as e:
                             # Should not happen if all_hashes is not empty and contains valid hashes, but defensive
                             action_taken = 'error_represent_hash'
                             message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [!] 错误 (确定代表哈希异常: {e})"
                             identified_errors += 1
                             representative_hash_object = None # Indicate we failed to get a representative hash


                        if representative_hash_object is not None:
                            is_duplicate = False
                            matching_canonical_relative_path = None # Store the *relative* path of the first encountered duplicate

                            # Check if the representative hash object is already in our map
                            if representative_hash_object in canonical_hashes_map:
                                is_duplicate = True
                                matching_canonical_relative_path = canonical_hashes_map[representative_hash_object]


                            if is_duplicate:
                                action_taken = 'identified_duplicate'
                                identified_duplicates += 1
                                delete_msg_prefix = "标记删除" if try_run else "删除"
                                # Customize message based on log_level
                                if log_level == 0:
                                    # log_level 0: detailed message including the original duplicate file (relative path)
                                    message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: {delete_msg_prefix} (重复, 内容与 {matching_canonical_relative_path} 相同)"
                                else: # log_level 1: concise message
                                    message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: {delete_msg_prefix} (重复)"

                                if not try_run:
                                    try:
                                        os.remove(filepath)
                                    except OSError as e:
                                        # Error specifically during deletion
                                        message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [!] 错误 (删除失败: {e})"
                                        action_taken = 'error_delete'
                                        identified_errors += 1

                            else: # Not a duplicate (based on 8-way check)
                                action_taken = 'kept'
                                # Add the representative hash object to the map with the current file's relative path
                                canonical_hashes_map[representative_hash_object] = relative_filepath
                                # Only print 'kept' message for log_level 0
                                if log_level == 0:
                                    message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [+] 保留 (新内容)"
                        # else: representative_hash_object was None due to error, message already set

            except FileNotFoundError:
                 # This can happen if a file is deleted between Pass 1 and Pass 2
                 action_taken = 'error_not_found'
                 message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [!] 错误 (文件未找到或已删除)"
                 identified_errors += 1
                 # Do not increment total_images_processed_attempted again, it was already incremented
            except UnidentifiedImageError:
                action_taken = 'error_bad_format'
                message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [!] 错误 (无效图片格式或已损坏)"
                identified_errors += 1
                # Optionally delete corrupted files here:
                # if not try_run:
                #     try: os.remove(filepath)
                #     except Exception as rm_e:
                #          message_to_print += f" 删除失败: {rm_e}" # Add delete error info to message
                #          action_taken = 'error_delete_corrupted' # Maybe a specific error type for this
            except Exception as e:
                action_taken = 'error_other'
                message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [!] 错误 (处理异常: {e})"
                identified_errors += 1

            # If message_to_print was not set by a specific action (e.g., error), set it for general errors
            if message_to_print is None and action_taken in ('error_hash_failed', 'error_hash_partial'):
                 if action_taken == 'error_hash_failed':
                      message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [!] 错误 (无法计算8种方向哈希)"
                 elif action_taken == 'error_hash_partial':
                      message_to_print = f"({processed_count+1}/{total_potential_images}) {relative_filepath}: [!] 警告 (未获得有效哈希)"


            # Print message above the tqdm bar based on log_level and action
            if message_to_print:
                 if log_level == 0:
                      # Use t.write to print messages above the bar
                      t.write(message_to_print)
                 elif log_level == 1:
                      # Only print if it was an identified deletion or an error
                      error_action_types = ('error_hash_failed', 'error_hash_partial', 'error_not_found',
                                             'error_bad_format', 'error_other', 'error_delete', 'error_represent_hash')
                      if action_taken in ('identified_low_res', 'identified_duplicate') or action_taken in error_action_types:
                           t.write(message_to_print)
                 # log_level 2 prints nothing via t.write()


            # Add delay *after* processing the file and potentially writing the message for it
            if delay_seconds > 0:
                 time.sleep(delay_seconds)

            # tqdm automatically calls update(1) after each iteration in the 'with' statement
            # if total was provided.

    # tqdm bar is automatically closed by the 'with' statement

    # Print final report after the tqdm bar is finished
    total_identified_for_deletion = identified_low_res + identified_duplicates
    # Note: remaining_images counts files *that would remain* after cleaning, among those attempted
    # Files with errors were attempted but not successfully processed or deleted.
    remaining_images = total_images_processed_attempted - total_identified_for_deletion - identified_errors

    # Calculate percentages based on total_images_processed_attempted
    if total_images_processed_attempted > 0:
        low_res_pct = (identified_low_res / total_images_processed_attempted) * 100
        duplicates_pct = (identified_duplicates / total_images_processed_attempted) * 100
        to_delete_pct = (total_identified_for_deletion / total_images_processed_attempted) * 100
        errors_pct = (identified_errors / total_images_processed_attempted) * 100
        # Remaining percentage is based on the count remaining
        remaining_pct = (remaining_images / total_images_processed_attempted) * 100

        # Verification: identified_low_res + identified_duplicates + identified_errors + remaining_images = total_images_processed_attempted
        # Check sum for verification if needed: print(f"Sum check: {identified_low_res + identified_duplicates + identified_errors + remaining_images} vs {total_images_processed_attempted}")
    else:
        # Handle case where no files were processed to avoid division by zero
        low_res_pct = 0.0
        duplicates_pct = 0.0
        to_delete_pct = 0.0
        errors_pct = 0.0
        remaining_pct = 0.0


    print("\n--- 清理完成统计 ---")
    if try_run:
        print("(Try Run 模式: 以下数字表示将被处理的文件)")
    print(f"扫描到的符合条件的图片文件总数 (受深度限制): {total_potential_images}") # Number found in Pass 1 after depth filter
    print(f"尝试处理的图片文件总数: {total_images_processed_attempted}") # Number fed into tqdm loop
    print(f"识别出分辨率过低图片数: {identified_low_res} ({low_res_pct:.2f}%)")
    print(f"识别出重复图片数 (含8种方向): {identified_duplicates} ({duplicates_pct:.2f}%)") # Clarified duplicate check
    print(f"总计识别出待处理图片数: {total_identified_for_deletion} ({to_delete_pct:.2f}%)")
    print(f"处理过程中发生错误文件数: {identified_errors} ({errors_pct:.2f}%)")
    # Remaining count is based on the number that made it through checks successfully
    print(f"预计保留图片总数: {remaining_images} ({remaining_pct:.2f}%)")
    print("----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理目录下的图片文件，删除低分辨率和重复图片（含8种方向翻转/旋转）。") # Updated description
    parser.add_argument("root_dir", help="要清理的根目录路径。")
    parser.add_argument("-r", "--resolution", required=True,
                        help="最小分辨率，格式为 宽度x高度 (例如: 800x600)。")
    parser.add_argument("-l", "--log-level", type=int, default=0, choices=[0, 1, 2],
                        help="打印信息等级 (0: 全部文件处理信息+进度条, 1: 简洁文件处理信息+进度条, 2: 只打印进度条)。默认: 0")
    # Using -t as alias for --try-run
    parser.add_argument("-t", "--try-run", action="store_true",
                        help="只检查文件并报告将被处理的文件，不实际删除。")
    # Add the delay parameter
    parser.add_argument("-d", "--delay", type=float, default=0.0,
                        help="处理每个文件后模拟延时的秒数 (浮点数, 例如 0.1)。用于观察进度条。默认: 0.0")
    # Add the max_depth parameter
    parser.add_argument("-depth", "--max-depth", type=int, default=-1,
                        help="限制扫描子目录的层级深度 (0: 仅根目录, 1: 根目录及其一级子目录, -1: 不限制)。默认: -1")


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

    # Normalize root_dir path for consistent relative path calculation
    args.root_dir = os.path.abspath(args.root_dir)


    # Execute cleaning
    try:
        clean_images(args.root_dir, min_res,
                     log_level=args.log_level, try_run=args.try_run,
                     delay_seconds=args.delay, max_depth=args.max_depth)
    except KeyboardInterrupt:
        print("\n清理过程被用户中断。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch general exceptions not handled inside the loop
        print(f"\n处理过程中发生未捕获的错误: {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc() # Optional: print full traceback for debugging
        sys.exit(1)