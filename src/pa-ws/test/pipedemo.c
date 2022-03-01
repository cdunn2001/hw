#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/wait.h>
#include <errno.h>

int status;
int pout[2];
int perr[2];
int pstat[2];
int main()
{

assert(pipe(pout) == 0);
assert(pipe(perr) == 0);
assert(pipe(pstat) == 0);

pid_t pid = fork();
if (pid == 0)
{   
    close(pout[0]);    // close reading end in the child
    close(perr[0]);    // close reading end in the child
    close(pstat[0]);   // close reading end in the child

#if 0
    // direct usage of file descriptors from C
    write(pout[1],"out\n",4);
    write(perr[1],"err\n",4);
    write(pstat[1],"stat\n",5);
    close(pout[1]);
    close(perr[1]);
    close(pstat[1]);
#elif 0
    // indirect usage of file descriptors with stdout/stderr rerouting using dup2
    dup2(pout[1], 1);  // send stdout to the pipe
    dup2(perr[1], 2);  // send stderr to the pipe
    fprintf(stdout,"out\n");
    fprintf(stderr,"err\n");
    write(pstat[1],"stat\n",5);
    close(pout[1]);
    close(perr[1]);
#else
    // use of numeric file descrptors from bash
    char line[128];
    const char* command = "/bin/bash";

#if 0
    // using system, which is inefficient because it hides another fork() under the hood
    sprintf(line,"%s -c 'echo stdout >&%d; echo stderr >&%d; echo stat >&%d'", command, pout[1], perr[1], pstat[1]);
    printf("here is the command: %s\n",line);
    system(line);
    close(pout[1]);
    close(perr[1]);
    close(pstat[1]);
#else
    // the most efficient way of launching bash
    dup2(pout[1], 1);  // send stdout to the pipe
    dup2(perr[1], 2);  // send stderr to the pipe
    sprintf(line,"echo stdout ; echo stderr >&2; echo stat >&%d", pstat[1]);
    printf("here is the cmd:%s and the arg: %s\n",command, line);
    int s = execl(command, command, "-c", line, 0);
    printf("execl failed with %d, errno %d\n",s,errno);
    perror("errno");
#endif
#endif
}
else
{
    // parent ...
    close(pout[1]);    // close reading end in the child
    close(perr[1]);    // close reading end in the child
    close(pstat[1]);    // close reading end in the child

    waitpid(pid, &status, 0);
    char buf[128];

    int l0 = read(pout[0], buf,sizeof(buf));
    if (l0 >= 0)
    {
    buf[l0]=0;
    printf("l0=%d pout=\"%s\"\n",l0,buf);
    }

    int l1 = read(perr[0], buf,sizeof(buf));
    if (l1 >=0)
    {
    buf[l1]=0;
    printf("l1=%d perr=\"%s\"\n",l1,buf);
    }

    int l2 = read(pstat[0], buf,sizeof(buf));
    if (l1 >=0)
    {
        buf[l2]=0;
        printf("l2=%d pstat=\"%s\"\n",l2,buf);
    }

    close(pout[0]);
    close(perr[0]);
    close(pstat[0]);
}

}
