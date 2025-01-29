from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import bios
import os
import anndata
import pandas as pd
import shutil
from fedscgen.FedScGen import FedScGen
from fedscgen.utils import aggregate, aggregate_batch_sizes

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'


@app_state('initial')
class InitialState(AppState):
    config = None
    model = None
    global_model = None
    global_weights = None

    def register(self):
        self.register_transition('Local Training', label='Start training')
        self.register_transition('Local Batch Sizes', label='Start correction')

    def run(self):
        self.config = bios.read(os.path.join(INPUT_DIR, 'config.yaml'))['fedscgen']
        self.store('config', self.config)
        kwargs = self.read_data()
        workflow = self.config['workflow']
        self.store('workflow', workflow)
        assert workflow in ['train', 'correction'], f'Invalid workflow type: {workflow}'
        if workflow == 'train':
            self.model = FedScGen(**kwargs)
            self.store('model', self.model)
            if self.is_coordinator:
                self.global_model = FedScGen(init_model_path=self.config['model']['init_model'], **kwargs)
                self.global_weights = self.global_model.model.state_dict()
                self.global_model.is_trained_ = True
                self.global_model.model.eval()
                self.broadcast_data(self.global_weights, memo='init_weights', send_to_self=False)
                self.store('global_weights', self.global_weights)
            return 'Local Training'
        else:
            self.model = FedScGen(init_model_path=self.config['model']['init_model'], **kwargs)
            self.store('model', self.model)
            return 'Local Batch Sizes'

    def read_data(self):
        shutil.copy2(os.path.join(INPUT_DIR, self.config['data']['adata']), os.path.join(OUTPUT_DIR, self.config['data']['adata']))
        adata = anndata.read_h5ad(os.path.join(INPUT_DIR, self.config['data']['adata']))
        self.store('adata', adata)
        hidden_sizes = [int(num) for num in self.config['model']['hidden_layer_sizes'].split(",")]
        kwargs = {"ref_model_path": self.config['model']['ref_model'],
                  "adata": adata,
                  "hidden_layer_sizes": hidden_sizes,
                  "z_dimension": self.config['model']['z_dimension'],
                  "batch_key": self.config['data']['batch_key'],
                  "cell_key": self.config['data']['cell_key'],
                  "lr": self.config['train']['lr'],
                  "epoch": self.config['train']['epoch'],
                  "batch_size": self.config['train']['batch_size'],
                  "stopping": {**self.config['train']['early_stopping']},
                  "overwrite": False,
                  "device": 'cpu'
                  }
        return kwargs


@app_state('Local Training')
class LocalTraining(AppState):
    n_rounds = 0
    model = None

    def register(self):
        self.register_transition('Model Aggregation', role=Role.COORDINATOR, label='Collect local updates')
        self.register_transition('Local Training', role=Role.PARTICIPANT, label='Next round of training')
        self.register_transition('Local Batch Sizes', role=Role.PARTICIPANT, label='Start correction')

        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        if self.model is None:
            self.model = self.load('model')
        self.n_rounds += 1
        finished = False
        if self.is_coordinator:
            global_weights = self.load('global_weights')
        else:
            if self.n_rounds == 1:
                global_weights = self.await_data(memo='init_weights')
            else:
                global_weights, finished = self.await_data(memo=f'global_weights_{self.n_rounds}')
        if finished:
            self.model.set_weights(global_weights)
            self.store('model', self.model)
            return 'Local Batch Sizes'

        self.update(f"Round {self.n_rounds} of training ...")
        local_updates = self.model.local_update(global_weights)
        self.send_data_to_coordinator(local_updates, memo=f'local_updates_{self.n_rounds}')

        if self.is_coordinator:
            return 'Model Aggregation'
        return 'Local Training'


@app_state('Model Aggregation', role=Role.COORDINATOR)
class ModelAggregation(AppState):
    n_rounds = 0

    def register(self):
        self.register_transition('Local Training', role=Role.COORDINATOR, label='Updated weights')
        self.register_transition('Local Batch Sizes', role=Role.COORDINATOR, label='Start correction')

    def run(self):
        self.n_rounds += 1
        clients_data = self.gather_data(memo=f'local_updates_{self.n_rounds}')
        local_weights = [client_data[0] for client_data in clients_data]
        local_n_samples = [client_data[1] for client_data in clients_data]
        self.log(f"Round {self.n_rounds} of local aggregating weights...")
        global_weights = aggregate(local_weights, local_n_samples)
        finish = self.n_rounds == self.load('config')['train']['n_rounds']
        self.broadcast_data([global_weights, finish], memo=f'global_weights_{self.n_rounds + 1}', send_to_self=False)
        if finish:
            return 'Local Batch Sizes'
        return 'Local Training'


@app_state('Local Batch Sizes')
class LocalBatchSizes(AppState):
    def register(self):
        self.register_transition('Dominant Batches', role=Role.COORDINATOR, label='Collect batch sizes')
        self.register_transition('Latent Genes', role=Role.PARTICIPANT, label='Wait for dominant batches')

    def run(self):
        model = self.load('model')
        batch_sizes = model.find_batch_size()
        self.send_data_to_coordinator([self.id, batch_sizes], memo='Local_Batch_Sizes')
        self.store('model', model)
        if self.is_coordinator:
            return 'Dominant Batches'
        return 'Latent Genes'


@app_state('Dominant Batches', role=Role.COORDINATOR)
class DominantBatches(AppState):
    def register(self):
        self.register_transition('Latent Genes', role=Role.COORDINATOR, label='Broadcast identified dominant batches')

    def run(self):
        batch_sizes = self.gather_data(memo='Local_Batch_Sizes', )
        batch_sizes = {item[0]: item[1] for item in batch_sizes}
        global_cell_sizes = aggregate_batch_sizes(batch_sizes)
        self.store('dominant_cell_types', global_cell_sizes.pop(self.id))
        for client_id, cell_types in global_cell_sizes.items():
            self.send_data_to_participant(cell_types, client_id, memo='dominant_cell_types')
        return 'Latent Genes'


@app_state('Latent Genes')
class MeanLatentGenes(AppState):
    def register(self):
        self.register_transition('Aggregate Latent Genes', role=Role.COORDINATOR, label='Collect MLGs')
        self.register_transition('Write Results', role=Role.PARTICIPANT, label='Wait for MLGs')

    def run(self):
        if self.is_coordinator:
            dominant_cell_types = self.load('dominant_cell_types')
        else:
            dominant_cell_types = self.await_data(memo='dominant_cell_types')
        model = self.load('model')
        mean_latent_genes = model.avg_local_cells(dominant_cell_types)
        self.send_data_to_coordinator(mean_latent_genes, memo='mean_latent_genes')
        if self.is_coordinator:
            return 'Aggregate Latent Genes'
        return 'Write Results'


@app_state('Aggregate Latent Genes', role=Role.COORDINATOR)
class AggregateLatentGenes(AppState):
    def register(self):
        self.register_transition('Write Results', role=Role.COORDINATOR, label='Correct local data')

    def run(self):
        mean_latent_genes = self.gather_data(memo='mean_latent_genes')
        global_mean_latent_genes = {}
        for c in mean_latent_genes:
            global_mean_latent_genes.update(c)
        self.broadcast_data(global_mean_latent_genes, memo='global_mean_latent_genes', send_to_self=False)
        self.store('global_mean_latent_genes', global_mean_latent_genes)
        return 'Write Results'


@app_state('Write Results')
class WriteResults(AppState):
    def register(self):
        self.register_transition('terminal', label='Finish')

    def run(self):
        model = self.load('model')
        if self.is_coordinator:
            global_mean_latent_genes = self.load('global_mean_latent_genes')
        else:
            global_mean_latent_genes = self.await_data(memo='global_mean_latent_genes')
        if self.load('config')['workflow'] == 'train':
            model.save(os.path.join(OUTPUT_DIR, "trained_model"))
        corrected = model.remove_batch_effect(global_mean_latent_genes)
        corrected.write(os.path.join(OUTPUT_DIR, "corrected.h5ad"))
        columns = list(range(1, self.load('config')['model']['z_dimension'] + 1))
        df = pd.DataFrame(data=list(global_mean_latent_genes.values()),
                          columns=columns,
                          index=list(global_mean_latent_genes.keys()))
        df.to_csv(os.path.join(OUTPUT_DIR, "GLMs.csv"), sep=",", index=True)
        return 'terminal'
