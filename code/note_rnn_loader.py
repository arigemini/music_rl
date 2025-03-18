from magenta.models.rl_tuner import rl_tuner_ops
from magenta.models.rl_tuner.note_rnn_loader import NoteRNNLoader as BaseNoteRNNLoader

class NoteRNNLoader(BaseNoteRNNLoader):
    def get_variable_name_dict(self):
        var_dict = dict()
        for var in self.variables():
            inner_name = rl_tuner_ops.get_inner_scope(var.name)
            inner_name = rl_tuner_ops.trim_variable_postfixes(inner_name)
            if '/Adam' in var.name:
                pass
            else:
                scope = self.checkpoint_scope + '/' + inner_name
                if scope == 'rnn_model/rnn/multi_rnn_cell/cell_0/lstm_cell/bias':
                    var_dict['rnn_model/RNN/MultiRNNCell/Cell0/LSTMCell/B'] = var
                elif scope == 'rnn_model/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel':
                    var_dict['rnn_model/RNN/MultiRNNCell/Cell0/LSTMCell/W_0'] = var
                else:
                    var_dict[scope] = var
        return var_dict