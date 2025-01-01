install.packages('nptest')
library(nptest)
subset_tests <- list("HJ",
                     "KC",
                     "SW",
                     "TB",
                     "FL",
                     "MA",
                     "OS",
                     "DS")
complete_tests <- list("perm",
                       "flip",
                       "both")
tests_name <- list(
  "Huh-Jhun",
  "Kennedy-Cade",
  "Still-White",
  "ter Braak",
  "Freedman-Lane",
  "Manly",
  "O’Gorman-Smith",
  "Draper-Stoneman"
)
data_path <-
  "testing_data"

homosceds <- list(TRUE, FALSE)

results <- data.frame(
  statistic = numeric(0),
  p_value = numeric(0),
  homosced = character(0),
  method = character(0),
  complete = character(0),
  sample = numeric(0),
  variate = character(0)
)

for (n in c(10, 15, 20)) {
  #### 1) univariate_subset ####
  uni_sub_x <- rnorm(n)
  uni_sub_z <- rnorm(n)
  uni_sub_y <- 3 + 2 * uni_sub_z + rnorm(n)
  # Make a data frame
  df_uni_sub <- data.frame(x = uni_sub_x,
                           z = uni_sub_z,
                           y = uni_sub_y)
  # Save as CSV
  write.csv(df_uni_sub,
            file.path(data_path, paste0("univariate_subset_n", n, ".csv")),
            row.names = FALSE)
  #### 2) multivariate_subset ####
  multi_sub_x <- rnorm(n)
  multi_sub_z <- rnorm(n)
  multi_sub_y <- cbind(1 + 3 * multi_sub_z + rnorm(n),
                       2 + 2 * multi_sub_z + rnorm(n),
                       3 + 1 * multi_sub_z + rnorm(n))
  # Make a data frame
  df_multi_sub <- data.frame(
    x  = multi_sub_x,
    z  = multi_sub_z,
    y1 = multi_sub_y[, 1],
    y2 = multi_sub_y[, 2],
    y3 = multi_sub_y[, 3]
  )
  # Save as CSV
  write.csv(df_multi_sub,
            file.path(data_path, paste0("multivariate_subset_n", n, ".csv")),
            row.names = FALSE)
  #### 6) run complete tests ####
  for (method in subset_tests) {
    for (homosced in homosceds) {
      # Perform the test
      return_obj <-
        np.reg.test(uni_sub_x,
                    uni_sub_y,
                    uni_sub_z,
                    homosced = homosced,
                    method = method)

      # Extract statistic and p_value
      statistic <- return_obj$statistic
      p_value <- return_obj$p.value

      # Store the results in the data frame
      results <- rbind(
        results,
        data.frame(
          statistic = statistic,
          p_value = p_value,
          homosced = homosced,
          method = method,
          complete = FALSE,
          variate = 'uni',
          sample=n

        )
      )
      return_obj <-
        np.reg.test(
          multi_sub_x,
          multi_sub_y,
          multi_sub_z,
          homosced = homosced,
          method = method
        )

      # Extract statistic and p_value
      statistic <- return_obj$statistic
      p_value <- return_obj$p.value

      # Store the results in the data frame
      results <- rbind(
        results,
        data.frame(
          statistic = statistic,
          p_value = p_value,
          homosced = homosced,
          method = method,
          complete = FALSE,
          variate = 'multi',
          sample=n
        )
      )
    }
  }
  #### 4) univariat complete ####
  univ_x <- cbind(rnorm(n), rnorm(n))
  univ_y <- rnorm(n)
  # Make a data frame
  df_univ <- data.frame(x1 = univ_x[, 1],
                        x2 = univ_x[, 2],
                        y  = univ_y)

  # Save as CSV
  write.csv(df_univ,
            file.path(data_path, paste0("univariate_n", n, ".csv")),
            row.names = FALSE)


  #### 5) multivariate complete ####
  multi_x <- cbind(rnorm(n), rnorm(n))
  multi_y <- matrix(rnorm(n * 3), nrow = n, ncol = 3)

  # Make a data frame
  df_multi <- data.frame(
    x1 = multi_x[, 1],
    x2 = multi_x[, 2],
    y1 = multi_y[, 1],
    y2 = multi_y[, 2],
    y3 = multi_y[, 3]
  )

  # Save as CSV
  write.csv(df_multi,
            file.path(data_path, paste0("multivariate_n", n, ".csv")),
            row.names = FALSE)
  #### 6) run complete tests ####
  for (method in complete_tests) {
    for (homosced in homosceds) {
      # Perform the test
      return_obj <-
        np.reg.test(univ_x, univ_y, homosced = homosced, method = method)
      # Extract statistic and p_value
      statistic <- return_obj$statistic
      p_value <- return_obj$p.value
      # Store the results in the data frame
      results <- rbind(
        results,
        data.frame(
          statistic = statistic,
          p_value = p_value,
          homosced = homosced,
          method = method,
          complete = TRUE,
          variate = 'uni',
          sample=n

        )
        )
        return_obj <-
          np.reg.test(
            multi_x,
            multi_y,
            homosced = homosced,
            method = method
          )
        # Extract statistic and p_value
        statistic <- return_obj$statistic
        p_value <- return_obj$p.value
        # Store the results in the data frame
        results <-
          rbind(
            results,
            data.frame(
              statistic = statistic,
              p_value = p_value,
              homosced = homosced,
              method = method,
              complete = TRUE,
              variate = 'multi',
              sample=n
            ))
    }
  }

}


write.csv(
  results,
  "testing_data\\R_results.csv",
  row.names = FALSE
)
