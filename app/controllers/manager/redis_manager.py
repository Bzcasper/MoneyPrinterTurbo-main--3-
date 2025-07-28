import json
import time
from typing import Dict

import redis
from loguru import logger

from app.controllers.manager.base_manager import TaskManager
from app.models.schema import VideoParams
from app.services import task as tm

FUNC_MAP = {
    "start": tm.start,
    # 'start_test': tm.start_test
}


class RedisTaskManager(TaskManager):
    def __init__(self, max_concurrent_tasks: int, redis_url: str):
        # Create Redis connection pool for better performance
        self.redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=20,
            retry_on_timeout=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30
        )
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        super().__init__(max_concurrent_tasks)

    def create_queue(self):
        return "task_queue"

    def enqueue(self, task: Dict):
        try:
            task_with_serializable_params = task.copy()

            if "params" in task["kwargs"] and isinstance(
                task["kwargs"]["params"], VideoParams
            ):
                task_with_serializable_params["kwargs"]["params"] = task["kwargs"][
                    "params"
                ].dict()

            # 将函数对象转换为其名称
            task_with_serializable_params["func"] = task["func"].__name__
            
            # Add timestamp and TTL
            task_with_serializable_params["enqueued_at"] = int(time.time())
            
            # Use pipeline for atomic operations
            pipeline = self.redis_client.pipeline()
            pipeline.rpush(self.queue, json.dumps(task_with_serializable_params))
            pipeline.expire(self.queue, 86400)  # 24 hour TTL
            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            raise

    def dequeue(self):
        task_json = self.redis_client.lpop(self.queue)
        if task_json:
            task_info = json.loads(task_json)
            # 将函数名称转换回函数对象
            task_info["func"] = FUNC_MAP[task_info["func"]]

            if "params" in task_info["kwargs"] and isinstance(
                task_info["kwargs"]["params"], dict
            ):
                task_info["kwargs"]["params"] = VideoParams(
                    **task_info["kwargs"]["params"]
                )

            return task_info
        return None

    def is_queue_empty(self):
        return self.redis_client.llen(self.queue) == 0
