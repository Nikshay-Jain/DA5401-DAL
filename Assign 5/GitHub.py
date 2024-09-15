import subprocess

def push():
    # subprocess.run(['git', 'pull'])
    try:
        subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(['git', 'push', '-u', 'origin', 'master'], check=True)
    print("\nRepository Updated Successfully.")

def act(file_path, action, report):
    subprocess.run(f'git {action} "{file_path}"')
    subprocess.run(f'git commit -m "{report} {file_path}"')
    print(f"{report} and committed: {file_path}")

command = "git status"
c=0
try:
    output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    file_lines = output.strip().split('\n')

    print(file_lines)
    c=0
    if(file_lines[1][0:20] == "Your branch is ahead"):
        push()

    l = int(input("Enter no of commits you want to make, else -1 to update full repository\n"))
    if(l==-1 or l>len(file_lines)-7):
        l = len(file_lines)-7

    # for totally new files in a new folder
    if(file_lines[3][0:15] == "Untracked files"):
        for i in range(l):
            file_path = file_lines[i+5][1:]
            print(file_path)
            act(file_path, "add", "Added")
            c+=1

    # for updated/deleted files in an existing folder
    for i in range(l-c):
        status = file_lines[i+6][1:2]
        file_path = file_lines[i+6][13:]
        print(status)
        print(file_path)

        if status == 'd':
            act(file_path, "rm", "Deleted")
        elif status == 'm':
            act(file_path, "add", "Updated")
        else:
            break
        c+=1
    
    output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    file_lines = output.strip().split('\n')

    k = 8 if c > 0 else 3

    # for new files in an existing folder
    if(len(file_lines)>=k):
        if(file_lines[k][0]=='U'):
            for i in range(l-c):
                file_path = file_lines[i+k+2][1:]
                print(file_path)
                if(file_path==''):
                    break
                else:
                    act(file_path, "add", "Added")
    push()

except subprocess.CalledProcessError as e:
    print("Command execution failed.")
    print("Error:", e)