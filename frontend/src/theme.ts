/** Mapeia arquitetura/tarefa para a cor do tema da aula (igual aos cards do material). */
export function archTheme(arch: string): string {
  if (["perceptron", "adaline", "mlp"].includes(arch)) return "t-found";
  if (arch === "cnn") return "t-cnn";
  if (["rnn", "lstm", "gru"].includes(arch)) return "t-rnn";
  if (arch === "transformer") return "t-trans";
  return "t-ae"; // autoencoder, gan
}

export function taskTheme(task: string): string {
  if (task.startsWith("tabular")) return "t-found";
  if (task.startsWith("image")) return "t-cnn";
  if (task.startsWith("sequence")) return "t-rnn";
  return "t-ae";
}

export const TASK_LABEL: Record<string, string> = {
  tabular_classification: "classificação tabular",
  tabular_regression: "regressão tabular",
  image_classification: "imagens",
  sequence_regression: "série temporal",
  sequence_classification: "classificação de sequências",
  unsupervised: "não supervisionado",
};

export const ARCH_LABEL: Record<string, string> = {
  perceptron: "Perceptron",
  adaline: "Adaline",
  mlp: "MLP",
  cnn: "CNN",
  rnn: "RNN",
  lstm: "LSTM",
  gru: "GRU",
  transformer: "Transformer",
  autoencoder: "Autoencoder",
  gan: "GAN",
};
