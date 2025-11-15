import os
import sys
import af
import argparse

def parser_args():
    parser = argparse.ArgumentParser("Submit job argparser")
    parser.add_argument("jobname")

def submit_job(
        name: str,
        command: str,
        working_dir: str = os.environ.get("PIPELINE_TOOLS")) -> None:
    print(f"Working Dir: {working_dir}")
    print("CGRU_LOCATION =", os.environ.get("CGRU_LOCATION"))
    job = af.Job(name)
    #job.setMaxRunningTasks(15)
    #job.setHostsMask('.*')
    block = af.Block('train', 'torch')   # or custom service name you use
    #block.setNumeric(0, 0, 1)
    block.setWorkingDirectory(working_dir)
    #block.setCommand(command=command)
    block.setParser('generic')        # <— name of your parser class/file
    task = af.Task('training model')
    task.setCommand(command)
    block.tasks.append(task)
    job.blocks.append(block)
    #job.setCmdPost('echo "cleanup here"')
    # Print job JSON to verify
    job.output()
    job.send()

if __name__ == "__main__":
    submit_job('torch_train_job', 
               f"tools\\wrappers\\af_wrapper.bat python {" ".join(sys.argv[1:])}")
