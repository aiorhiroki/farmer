import os
import glob
import shutil
import mlflow
from mlflow.tracking import MlflowClient

class MlflowClientWrapper:
    _singleton = None
    _mlflow_client = None
    _run = None
    _run_id = None
    _experiment_id = None
    
    @classmethod
    def __internal_new__(cls, tracking_uri="", registry_uri="", experiment_name="", run_name="", user_name=""):
        """
        インスタンスを生成する
        MlflowClientの生成、experiment, runを開始する

        Args:
            experiment_name (str, optional): [description]. Defaults to "".
            run_name (str, optional): [description]. Defaults to "".
            user_name (str, optional): [description]. Defaults to "".

        Returns:
            cls
        """
        print('[I] MlflowClientWrapper new')
        
        cls._mlflow_client = MlflowClient(tracking_uri, registry_uri)
        # mlrunsのパスを指定
        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            cls._experiment_id = cls._mlflow_client.create_experiment(experiment_name)
            
        except mlflow.exceptions.MlflowException:
            # 既にexperimentが存在している場合はset_experimentを実行する
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            cls._experiment_id = experiment.experiment_id
            
        try: 
            cls._run = mlflow.start_run(experiment_id=cls._experiment_id, run_name=run_name)
            cls._run_id = cls._run.info.run_id
            cls.set_tags({"mlflow.user": user_name})
        except Exception:
            print(f'run {cls._run_id} already exist.')
        
        print(f'experiment_id: {cls._experiment_id}, run_id: {cls._run_id}')

        print('[O] MlflowClientWrapper new')
        return cls

    @classmethod
    def get_artifacts_path(cls):
        return os.path.join(mlflow.get_tracking_uri(), cls._experiment_id, cls._run_id, 'artifacts')

    @classmethod
    def create_run(cls, **kwargs):
        """
        インスタンスを生成・初期化処理を行う
        ２回目以降はインスタンスを生成しない。
        """
        if not cls._singleton:
            cls._singleton = cls.__internal_new__(**kwargs)
        return cls._singleton
    
    @classmethod
    def end_run(cls):
        """
        runを停止し、インスタンスを解放する
        NOTE:
            連続学習する場合に同じインスタンスを使い回してしまうとmlflowで同じrun idを指定してしまうためエラーになる
            学習が終了したら必ずend_runを実行する
        """
        if cls._singleton:
            mlflow.end_run()
            cls._singleton = None
            cls._mlflow_client = None
            cls._run = None
            cls._run_id = None
            cls._experiment_id = None
        else:
            print('MlflowClientWrapper instance is already None.')

    @classmethod
    def get_instance(cls):
        return cls._singleton
    
    @classmethod
    def is_running(cls):
        if cls._run:
            return True
        else:
            return False
    
    @classmethod
    def save_artifacts_to_mlruns(cls, src_folder_path, artifact_dir_name=""):
        print("[I] save_artifacts_to_mlruns")
        dst_dir_path = os.path.join(cls.get_artifacts_path(), artifact_dir_name)
        os.makedirs(dst_dir_path, exist_ok=True)
        
        files = glob.glob(os.path.join(src_folder_path, '*'))
        for f in files:
            if os.path.isfile(f):
                print(f'{f} -> {dst_dir_path}')
                shutil.copy(f, dst_dir_path)
        print("[O] save_artifacts_to_mlruns")

    @classmethod
    def save_artifact_to_mlruns(cls, src_item_path, artifact_dir_name=""):
        print("[I] save_artifact_to_mlruns")
        dst_dir_path = os.path.join(cls.get_artifacts_path(), artifact_dir_name)
        print(f'{src_item_path} -> {dst_dir_path}')
        os.makedirs(dst_dir_path, exist_ok=True)
        shutil.copy(src_item_path, str(dst_dir_path))
        print("[O] save_artifact_to_mlruns")

    @classmethod
    def register_model(cls, model_name="model"):
        """
        mlflowにモデルを登録する
        NOTE: モデルをartifactに保存してから本メソッドを呼び出すこと
        TODO: 現状はURIの設定が不正だと言われてしまうため要検討

        Args:
            model_name (str): 登録するモデルの名称
        """
        print("[I] register_model")
        
        result = cls._mlflow_client.create_model_version(
            name=model_name,
            source=f"mlruns/{cls._experiment_id}/{cls._run_id}/artifacts/model",
            run_id=cls._run_id
        )
        print(f"[O] register_model: {result}")
    
    @classmethod
    def log_metrics(cls, metrics: dict):
        for metric_name, metric_val in metrics.items():
            cls._mlflow_client.log_metric(cls._run_id, metric_name, metric_val)

    @classmethod
    def log_metrics_with_array(cls, metrics: dict):
        for metric_name, metric_array in metrics.items():
            for i, metric_val in enumerate(metric_array):
                cls._mlflow_client.log_metric(cls._run_id, f"{metric_name}_{i}", metric_val)

    @classmethod
    def log_metrics_with_array_per_step(cls, metrics: dict, step: int):
        for metric_name, metric_array in metrics.items():
            for i, metric_val in enumerate(metric_array):
                cls._mlflow_client.log_metric(cls._run_id, f"{metric_name}_{i}", metric_val, step)

    @classmethod
    def log_params(cls, params: dict):
        for param_name, param_val in params.items():
            cls._mlflow_client.log_param(cls._run_id, param_name, param_val)
        
    @classmethod
    def set_tags(cls, tags: dict):
        for tag_name, tag_val in tags.items():
            cls._mlflow_client.set_tag(cls._run_id, tag_name, tag_val)