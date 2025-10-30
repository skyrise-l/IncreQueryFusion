#!/bin/bash



# 配置部分
PYTHON_SCRIPTS=(   
    "python -u -m launcher.main --dataset flight_mul --mode batch --fusion_model DART" 
    "python -u -m launcher.main --dataset flight_mul --mode batch --fusion_model CASE" 
    "python -u -m launcher.main --dataset flight_mul --mode batch --fusion_model TruthFinder" 

    "python -u -m launcher.main --dataset flight_mul --mode baseline --fusion_model DART" 
    "python -u -m launcher.main --dataset flight_mul --mode baseline --fusion_model CASE" 
    "python -u -m launcher.main --dataset flight_mul --mode baseline --fusion_model TruthFinder"   

    "python -u -m launcher.main --dataset flight_mul"

    "python -u -m launcher.main --dataset movie_mul --mode baseline --fusion_model DART" 
    "python -u -m launcher.main --dataset movie_mul --mode baseline --fusion_model CASE" 
    "python -u -m launcher.main --dataset movie_mul --mode baseline --fusion_model TruthFinder" 
)
LOG_DIR="./logs"
PID_FILE="./script_pids.pid"
PIDS=()

# 创建日志目录
mkdir -p "$LOG_DIR"

# 清理函数
cleanup() {
    echo "正在清理进程..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "已终止进程: $pid"
        fi
    done
    rm -f "$PID_FILE"
    exit
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

echo "开始执行Python脚本序列..."

# 生成日志文件名函数
get_log_filename() {
    local index=$1
    echo "script_${index}.log"
}

# 逐个执行Python脚本
for i in "${!PYTHON_SCRIPTS[@]}"; do
    script_cmd="${PYTHON_SCRIPTS[$i]}"
    
    # 生成日志文件名
    LOG_FILE=$(get_log_filename $i)
    LOG_PATH="$LOG_DIR/$LOG_FILE"
    
    echo "正在启动: $script_cmd"
    echo "日志文件: $LOG_PATH"
    
    # 在日志文件开头写入当前执行的命令
    echo "执行命令: $script_cmd" > "$LOG_PATH"
    echo "开始时间: $(date)" >> "$LOG_PATH"
    echo "========================================" >> "$LOG_PATH"
    
    # 执行命令并将输出追加到日志文件
    nohup $script_cmd >> "$LOG_PATH" 2>&1 &
    CURRENT_PID=$!
    PIDS+=("$CURRENT_PID")
    echo "$CURRENT_PID" >> "$PID_FILE"
    
    # 等待当前脚本完成
    echo "等待命令完成 (PID: $CURRENT_PID)..."
    wait $CURRENT_PID
    EXIT_CODE=$?
    
    # 在日志文件末尾记录执行结果
    echo "========================================" >> "$LOG_PATH"
    echo "结束时间: $(date)" >> "$LOG_PATH"
    echo "退出码: $EXIT_CODE" >> "$LOG_PATH"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "命令执行成功: $script_cmd"
    else
        echo "警告: 命令异常退出 (退出码: $EXIT_CODE): $script_cmd"
    fi
    
    # 从PID文件中移除已完成的进程
    sed -i "/^$CURRENT_PID$/d" "$PID_FILE"
    
    echo "----------------------------------------"
done

echo "所有脚本执行完成"
rm -f "$PID_FILE"