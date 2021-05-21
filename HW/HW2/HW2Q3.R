library("nleqslv")
rm(list=ls())

fct = function(t) {
  2 / t + 2 - 4 / (1 + exp(-t))
}

link = function(x, a, b) {
  t = b * (x - a)
  1 / (1 + exp(t)) / (1 + exp(-t))
}

inv_info = function(a, b) {
  ga = - 1.543405 / b + a
  matrix(c(1 / b^2, 0, 0, 1 / ((ga-a)^2)) / 0.82396 / 0.17604, ncol = 2)
}

sens = function(x, a, b) {
  ca = c(-b, x-a) * link(x, a, b) * (1 - link(x, a, b))
  biao = c(-b, x-a)
  t(ca) %*% inv_info(a, b) %*% ca - 2
}

hi = seq(-10, 10, 0.01)
output = c()
for (x in hi) {
  output <- c(output, sens(x, a=1, b=1))
}
plot(hi, output, type = "l", 
     main = "a=0.5, b=0.1", xlab = "x", ylab = "sensitivity")

which.max(output)
hi[1350]
