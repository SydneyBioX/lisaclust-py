## Shared test cases: lisaClust (R) computes the expected results; the Python tests (tests/test_shared_cases.py)
## must reproduce them. Rerun after any change to the core or to either front end:
##   Rscript tests/shared_cases/make_cases.R      (needs lisaClust >= 1.21.7)
suppressPackageStartupMessages(library(lisaClust))
out_dir <- file.path("tests", "shared_cases")
set.seed(20261009)
img <- function(id, n, shift) {
  x <- runif(n, 0, 300); y <- runif(n, 0, 200)
  type <- ifelse(x + shift * y > 150, "tumour", sample(c("T", "B", "macro"), n, TRUE, prob = c(0.5, 0.2, 0.3)))
  type[sample(n, n %/% 10)] <- "stroma"
  data.frame(cellID = paste0(id, "_", seq_len(n)), imageID = id, x = x, y = y, cellType = type)
}
cells <- rbind(img("a", 600, 0.3), img("b", 500, -0.4), img("c", 400, 0.8))
cells$cellType <- factor(cells$cellType, levels = unique(cells$cellType))
write.csv(cells, file.path(out_dir, "cells.csv"), row.names = FALSE)
# both packages compute from the file as written (15 significant digits)
cells <- read.csv(file.path(out_dir, "cells.csv"))
cells$cellType <- factor(cells$cellType, levels = unique(cells$cellType))
cases <- list(
  square_K = list(r = c(10, 25, 50), window = "square", lisaFunc = "K"),
  square_L = list(r = c(15, 40), window = "square", lisaFunc = "L"),
  convex_K = list(r = c(10, 25, 50), window = "convex", lisaFunc = "K"),
  large_r = list(r = c(20, 80, 500), window = "square", lisaFunc = "K")
)
for (nm in names(cases)) {
  a <- cases[[nm]]
  cv <- suppressMessages(lisa(cells, r = a$r, window = a$window, lisaFunc = a$lisaFunc))
  write.csv(data.frame(cellID = rownames(cv), cv, check.names = FALSE), file.path(out_dir, paste0(nm, ".csv")),
            row.names = FALSE)
}
# region outlines of image a (lisaClust's hatchingPlot): regions from the cell types
a <- cells[cells$imageID == "a", ]
a$region <- ifelse(a$cellType == "tumour", "r1", ifelse(a$cellType %in% c("T", "B"), "r2", "r3"))
write.csv(a[, c("cellID", "x", "y", "region")], file.path(out_dir, "outline_cells.csv"), row.names = FALSE)
a <- read.csv(file.path(out_dir, "outline_cells.csv"))
outlines <- lapply(c(square = "square", convex = "convex", concave = "concave"), function(w) {
  code <- as.numeric(factor(a$region))
  polys <- lisaClust:::regionPolygons(a$x, a$y, code, suppressMessages(lisaClust:::makeWindow(a, w, NULL)))
  setNames(polys[seq_along(levels(factor(a$region)))], levels(factor(a$region)))
})
jsonlite::write_json(outlines, file.path(out_dir, "outlines.json"), digits = NA)

jsonlite::write_json(lapply(cases, function(a) list(r = a$r, window = a$window, lisa_func = a$lisaFunc)),
                     file.path(out_dir, "cases.json"), auto_unbox = TRUE, pretty = TRUE)
cat("wrote", length(cases), "cases with lisaClust", as.character(packageVersion("lisaClust")), "\n")
