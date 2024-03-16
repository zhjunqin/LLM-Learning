# Nsight systems 

https://docs.nvidia.com/nsight-systems/UserGuide/index.html

## CLI

使用帮助

```
# nsys help
 usage: nsys [--version] [--help] <command> [<args>] [application] [<application args>]

 The most commonly used nsys commands are:
        profile       Run an application and capture its profile into a nsys-rep file.
        launch        Launch an application ready to be profiled.
        start         Start a profiling session.
        stop          Stop a profiling session and capture its profile into a nsys-rep file.
        cancel        Cancel a profiling session and discard any collected data.
        service       Launch the Nsight Systems data service.
        stats         Generate statistics from an existing nsys-rep or SQLite file.
        status        Provide current status of CLI or the collection environment.
        shutdown      Disconnect launched processes from the profiler and shutdown the profiler.
        sessions list List active sessions.
        export        Export nsys-rep file into another format.
        analyze       Identify optimization opportunities in a nsys-rep or SQLITE file.
        recipe        Run a recipe for multi-node analysis.
        nvprof        Translate nvprof switches to nsys switches and execute collection.

```

| 命令    | 描述                                                         |
| ------- | ------------------------------------------------------------ |
| profile | 运行一个应用程序并将其性能分析数据保存到一个nsys-rep文件中。 |
| analyze | 分析 nsys-rep 或 SQLITE 文件中定位有没有可以优化的地方。     |
| stats   | 从现有的 nsys-rep 或 SQLite 文件生成统计数据。               |


### 示例

```
nsys profile --stats=true --sample=cpu --trace=cuda,cudnn,cublas,nvtx,osrt,oshmem  ./my_bin

nsys analyze report1.nsys-rep

nsys stats report1.sqlite
```



### Jetson 如何在容器中使用 nsys

https://forums.developer.nvidia.com/t/nsight-systems-profiling-in-l4t-container/191072

mount the Nsight Systems folder to enable access

```
$ docker run -it --rm --net=host --runtime nvidia -v /opt/nvidia/nsight-systems-cli/:/opt/nvidia/nsight-systems-cli nvcr.io/nvidia/l4t-base:r32.5.0
```

