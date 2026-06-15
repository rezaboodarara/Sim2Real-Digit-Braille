import subprocess
import os
import shutil
import glob

def copy_and_rename(source_path, target_folder, new_filename):
    """
    Copies a file to a target folder with a new name.
    """
    # Check if the source file exists
    if not os.path.isfile(source_path):
        print(f"Error: The file '{source_path}' does not exist.")
        return

    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created target folder: {target_folder}")

    # Construct the full destination path
    destination_path = os.path.join(target_folder, new_filename)

    try:
        # Perform the copy
        shutil.copy2(source_path, destination_path)
        #print(f"Success! Copied '{source_path}' to '{destination_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")



#conf
letters = 'nopq'

#source_path = '~/Desktop/tacto/examples/objects/samples/letter-a/letter-a.obj'

for letter in letters:
    source_folder = '~/Desktop/tacto/examples/objects/samples'
    target_letter = f'letter-{letter}'
    source_path = os.path.expanduser(f'{source_folder}/{target_letter}/{target_letter}100.obj')
    target_folder = './objects'
    new_filename = 'Template.obj'
    total_files_number = len(os.listdir(os.path.expanduser(f'{source_folder}/{target_letter}')))//2
    #target_object_file = '/objects/samples/letter-a/'
    
    for file_number in range(0,total_files_number):
        if (file_number == 0):
            source_path = os.path.expanduser(f'{source_folder}/{target_letter}/{target_letter}.obj')
        else:
            source_path = os.path.expanduser(f'{source_folder}/{target_letter}/{target_letter}{file_number}.obj')
        copy_and_rename(source_path, target_folder, new_filename)

        #run the tacto
        subprocess.run(["python", "Template.py"])
        
        
        output_file_rgb =  '/home/reza/Desktop/tacto/examples/tactile_output.png'
        output_file_depth = '/home/reza/Desktop/tacto/examples/depth_output.png'
        output_target_rgb = os.path.expanduser(f'~/Desktop/tacto/examples/objects/output/rgb/letter-{letter}')
        output_target_depth = os.path.expanduser(f'~/Desktop/tacto/examples/objects/output/depth/letter-{letter}')
        
        output_name_rgb = f'{letter.capitalize()}-{file_number}.png'
        output_name_depth = f'{letter.capitalize()}-{file_number}.png'
        
        copy_and_rename(output_file_rgb, output_target_rgb, output_name_rgb)
        copy_and_rename(output_file_depth, output_target_depth, output_name_depth)
        print(letter, ':', file_number, 'generated')
    print(f"{target_letter} done.")
