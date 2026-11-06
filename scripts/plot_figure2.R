#!/usr/bin/env Rscript

# Replot of the paper's original 0502.R Figure 2 layout.
# Render the selected final source table without replacing model rows.
options(stringsAsFactors = FALSE, scipen = 999)
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr); library(stringr)
  library(forcats); library(scales); library(cowplot); library(ggrepel)
})

src_path <- Sys.getenv("FIGURE2_SOURCE", unset = "")
out_dir <- Sys.getenv("FIGURE2_OUT", unset = "")
if (!nzchar(src_path) || !nzchar(out_dir)) stop("Set FIGURE2_SOURCE and FIGURE2_OUT explicitly")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
font_family <- "sans"  # one family for every panel and label; Cairo embeds DejaVu Sans

series_palette <- c(
  "Observed" = "#CA6E28",
  "Mosquito-Graphormer" = "#005596",
  "TCN" = "#C0504D",
  "Graphormer combined with SEIR" = "#7BA6D8"
)
muted_risk_gradient <- c("#234A73", "#2F6F78", "#5F7652", "#946134", "#873448")
city_to_english <- c(
  "佛山市"="Foshan", "广州市"="Guangzhou", "深圳市"="Shenzhen", "东莞市"="Dongguan",
  "南宁市"="Nanning", "珠海市"="Zhuhai", "百色市"="Baise", "贺州市"="Hezhou",
  "湛江市"="Zhanjiang", "泉州市"="Quanzhou", "潮州市"="Chaozhou", "惠州市"="Huizhou",
  "贵港市"="Guigang", "梧州市"="Wuzhou", "桂林市"="Guilin", "云浮市"="Yunfu"
)
city_label <- function(x) { y <- unname(city_to_english[x]); y[is.na(y)] <- gsub("市$", "", x[is.na(y)]); y }

fmt_p <- function(p) ifelse(is.na(p), "NA", ifelse(p < 0.001, "< 0.001", sprintf("= %.4f", p)))
fmt_r <- function(r) ifelse(is.na(r), "NA", sprintf("%.2f", r))
safe_cor_test <- function(x, y) {
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]
  if (length(x) < 3 || sd(x) == 0 || sd(y) == 0) return(list(estimate=NA_real_, p.value=NA_real_))
  ct <- suppressWarnings(cor.test(x, y, method="pearson")); list(estimate=unname(ct$estimate), p.value=ct$p.value)
}
fmt_day_month <- function(x) {
  sprintf("%d %s", as.integer(format(x, "%d")), format(x, "%b"))
}
display_x_label <- function(x) {
  y <- as.character(x)
  if (grepl("^\\d{4}-\\d{2}-\\d{2}$", y)) {
    return(fmt_day_month(as.Date(y)))
  }
  if (grepl("^\\d{2}\\.\\d{2}-\\d{2}\\.\\d{2}$", y)) {
    a <- as.Date(paste0("2025.", substr(y, 1, 5)), format="%Y.%m.%d")
    b <- as.Date(paste0("2025.", substr(y, 7, 11)), format="%Y.%m.%d")
    if (format(a, "%b") == format(b, "%b")) return(sprintf("%d-%d %s", as.integer(format(a, "%d")), as.integer(format(b, "%d")), format(b, "%b")))
    return(sprintf("%s-%s", fmt_day_month(a), fmt_day_month(b)))
  }
  y
}

theme_lancet_bar_clean <- function(base_size=14.2) {
  theme_bw(base_size=base_size, base_family=font_family) + theme(
    panel.background=element_rect(fill="white", color=NA), plot.background=element_rect(fill="white", color=NA),
    panel.grid.major.x=element_blank(), panel.grid.minor=element_blank(),
    panel.grid.major.y=element_line(color="#E5E5E5", linewidth=.26), panel.border=element_blank(),
    axis.line.x.bottom=element_line(color="black", linewidth=.45), axis.line.y.left=element_line(color="black", linewidth=.45),
    axis.line.x.top=element_blank(), axis.line.y.right=element_blank(), axis.title=element_text(color="black", size=16),
    axis.text=element_text(color="black", size=13.4), axis.text.x=element_text(angle=42, hjust=1, vjust=1),
    axis.ticks=element_line(color="black", linewidth=.30), legend.position="top", legend.title=element_blank(),
    legend.text=element_text(size=14.2, margin=margin(t=0,r=13,b=0,l=2)), legend.key.size=unit(.56,"cm"),
    legend.spacing.x=unit(.04,"cm"), plot.title=element_blank(), plot.subtitle=element_blank(), plot.caption=element_blank(), plot.margin=margin(6,6,6,6)
  )
}
theme_lancet_like_clean <- function(base_size=14.2) {
  theme_bw(base_size=base_size, base_family=font_family) + theme(
    panel.background=element_rect(fill="white", color=NA), plot.background=element_rect(fill="white", color=NA),
    panel.grid.major.x=element_blank(), panel.grid.minor=element_blank(), panel.grid.major.y=element_line(color="#E5E5E5", linewidth=.26),
    panel.border=element_blank(), axis.line.x.bottom=element_line(color="black", linewidth=.45), axis.line.y.left=element_line(color="black", linewidth=.45),
    axis.line.x.top=element_blank(), axis.line.y.right=element_blank(), axis.title=element_text(color="black", size=15.8),
    axis.text=element_text(color="black", size=13.4), axis.ticks=element_line(color="black", linewidth=.30), legend.position="none",
    plot.title=element_blank(), plot.subtitle=element_blank(), plot.caption=element_blank(), plot.margin=margin(6,6,6,6)
  )
}
theme_corr_clean <- function(base_size=14.2) {
  theme_bw(base_size=base_size, base_family=font_family) + theme(
    panel.background=element_rect(fill="white", color=NA), plot.background=element_rect(fill="white", color=NA), panel.grid=element_blank(), panel.border=element_blank(),
    axis.line.x.bottom=element_line(color="black", linewidth=.45), axis.line.y.left=element_line(color="black", linewidth=.45), axis.line.x.top=element_blank(), axis.line.y.right=element_blank(),
    axis.title=element_text(color="black", size=16), axis.text=element_text(color="black", size=13.4), axis.ticks=element_line(color="black", linewidth=.30),
    legend.position="right", legend.title=element_text(size=11.8, face="bold"), legend.text=element_text(size=10.8), legend.key.height=unit(.42,"cm"), legend.key.width=unit(.34,"cm"),
    plot.title=element_blank(), plot.subtitle=element_blank(), plot.caption=element_blank(), plot.margin=margin(6,6,6,6)
  )
}

save_all <- function(p, stem, w=1396/72, h=1044/72) {
  ggsave(file.path(out_dir,paste0(stem,".pdf")), p, width=w, height=h, units="in", device=cairo_pdf, bg="white")
  ggsave(file.path(out_dir,paste0(stem,".png")), p, width=w, height=h, units="in", dpi=300, bg="white")
  ggsave(file.path(out_dir,paste0(stem,".svg")), p, width=w, height=h, units="in", device=grDevices::svg, bg="white")
}

src <- read_tsv(src_path, show_col_types=FALSE, progress=FALSE)
src$date_or_window <- as.character(src$date_or_window)

bar_df <- function(panel_name, city=NULL, models=c("Observed","Mosquito-Graphormer")) {
  z <- src %>% filter(.data$panel==.env$panel_name, .data$model %in% .env$models)
  if (!is.null(city)) z <- z %>% filter(.data$city==.env$city)
  z %>% mutate(series=factor(model, levels=models), value=if_else(model=="Observed", observed, predicted_mean), lo=pred_lower, hi=pred_upper, xlab=factor(date_or_window, levels=unique(date_or_window)), xdisplay=vapply(as.character(xlab), display_x_label, character(1)))
}
bar_plot <- function(df, ylab="Cumulative cases", title_text=NULL, legend=TRUE, title_y=NA_real_) {
  n <- nlevels(df$series); off <- if(n<=2) c(-.18,.18) else c(-.36,-.12,.12,.36); df$xpos <- as.numeric(df$xlab)+off[as.integer(df$series)]
  x_labels <- df %>% distinct(xlab, xdisplay) %>% arrange(as.numeric(xlab))
  p <- ggplot(df, aes(x=xpos,y=value,fill=series)) + geom_col(width=if(n<=2) .28 else .14, color="black", linewidth=.18) +
    geom_errorbar(data=df %>% filter(is.finite(lo)), aes(ymin=lo,ymax=hi), width=if(n<=2) .08 else .05, linewidth=.34, color="black") +
    scale_fill_manual(values=series_palette[levels(df$series)], drop=FALSE) + scale_x_continuous(breaks=seq_along(levels(df$xlab)), labels=x_labels$xdisplay, expand=expansion(mult=c(.015,.015))) +
    scale_y_continuous(labels=comma, expand=expansion(mult=c(0,.08))) + labs(x=NULL,y=ylab) + theme_lancet_bar_clean()
  if(!is.null(title_text)) {
    y_title <- if (is.na(title_y)) max(c(df$value, df$hi), na.rm=TRUE) * 1.08 else title_y
    title_vjust <- if (is.infinite(title_y)) 1.30 else 0.5
    p <- p + annotate("text", x=mean(seq_along(levels(df$xlab))), y=y_title, label=title_text, fontface="bold", size=5.1, family=font_family, vjust=title_vjust)
  }
  if(!legend) p <- p + theme(legend.position="none")
  p
}

# Original Figure 2 panel data
pA <- bar_plot(bar_df("a", "Foshan", c("Observed","Mosquito-Graphormer")), "Daily cases", legend=TRUE)
pB <- bar_plot(bar_df("b", "Foshan", c("Observed","Mosquito-Graphormer","TCN","Graphormer combined with SEIR")), legend=TRUE) + guides(fill=guide_legend(ncol=2, byrow=TRUE))

city_panel <- function(city, letter) {
  d <- bar_df("cde", city, c("Observed","Mosquito-Graphormer")); d$obs <- ifelse(d$series=="Observed",d$value,NA); d$pred <- ifelse(d$series=="Mosquito-Graphormer",d$value,NA)
  q <- d %>% group_by(xlab) %>% summarise(obs=max(obs,na.rm=TRUE), pred=max(pred,na.rm=TRUE), .groups="drop")
  ct <- safe_cor_test(q$pred,q$obs); title <- paste0(city," | 5-day forecast | Pearson r = ",fmt_r(ct$estimate))
  bar_plot(d, "Cumulative cases", title, legend=FALSE, title_y=Inf)
}
pC <- city_panel("Foshan", "c"); pD <- city_panel("Guangzhou", "d"); pE <- city_panel("Huizhou", "e")

# f: the original forest-plot layout (H5/H7 columns; RMSE/MAE rows).
fdat <- src %>% filter(panel=="f", model %in% c("Mosquito-Graphormer","TCN","Graphormer combined with SEIR"), is.finite(predicted_mean), !is.na(metric)) %>%
  transmute(model, horizon=str_replace(date_or_window,"H",""), metric=metric, mean=predicted_mean, lo=pred_lower, hi=pred_upper) %>% mutate(horizon=factor(horizon,levels=c("5","7")), metric=factor(metric,levels=c("RMSE","MAE")), model=factor(model,levels=c("Graphormer combined with SEIR","TCN","Mosquito-Graphormer")))
pF <- ggplot(fdat, aes(x=mean,y=model,color=model)) + geom_errorbar(aes(xmin=lo,xmax=hi), orientation="y", width=.18, linewidth=.55) + geom_point(size=3.05) + facet_grid(metric~horizon, scales="free_x", labeller=labeller(horizon=c("5"="5-day forecast","7"="7-day forecast"))) + scale_color_manual(values=series_palette[levels(fdat$model)], drop=FALSE) + labs(x="Forecast error (cases)",y=NULL) + theme_lancet_like_clean(base_size=14.2) + theme(axis.text.y=element_text(face="bold",size=13.6), axis.title.x=element_text(size=16), strip.text=element_text(face="bold",size=14.2))

make_corr <- function(panel_name, cumulative=FALSE) {
  z <- src %>% filter(.data$panel==panel_name, .data$model=="Mosquito-Graphormer") %>% transmute(city, x=predicted_mean, y=observed)
  z <- z %>% mutate(is_top10=rank(-x,ties.method="first")<=10, x_log=log1p(x), y_log=log1p(y), label=city_label(city))
  ct <- safe_cor_test(z$x_log,z$y_log); max_real <- max(c(z$x,z$y),na.rm=TRUE)*1.05; br <- c(0,1,5,10,25,50,100,250,500,1000); br <- br[br<=max_real]; if(length(br)<3) br <- pretty(c(0,max_real),n=4)
  title <- if(cumulative) "Predicted mean 5-day cumulative cases" else "Predicted mean daily cases"; ytitle <- if(cumulative) "Observed mean 5-day cases" else "Observed mean daily cases"; lg <- "Top 10\npredicted\ncases"
  ggplot(z,aes(x=x_log,y=y_log)) + geom_abline(intercept=0,slope=1,color="grey55",linetype="22",linewidth=.5) + geom_smooth(method="lm",se=TRUE,color="black",fill="grey84",linewidth=.65) + geom_point(data=z %>% filter(!is_top10),color="grey75",size=1.85,alpha=.62) + geom_point(data=z %>% filter(is_top10),aes(color=x),shape=16,size=3.75) + ggrepel::geom_text_repel(data=z %>% filter(is_top10),aes(label=label,color=x),size=5.15,fontface="bold",family=font_family,box.padding=.38,point.padding=.24,min.segment.length=0,segment.color="grey55",segment.size=.32,max.overlaps=Inf,seed=123,show.legend=FALSE) + annotate("text",x=log1p(max_real)*.05,y=log1p(max_real)*.96,label=paste0("Pearson r = ",fmt_r(ct$estimate),"\np ",fmt_p(ct$p.value)),hjust=0,vjust=1,size=4.8,fontface="bold",family=font_family) + scale_color_gradientn(colors=muted_risk_gradient,name=lg,breaks=pretty_breaks(n=3)) + scale_x_continuous(breaks=log1p(br),labels=comma(br),expand=expansion(mult=c(.01,.01))) + scale_y_continuous(breaks=log1p(br),labels=comma(br),expand=expansion(mult=c(.01,.01))) + labs(x=title,y=ytitle) + coord_cartesian(xlim=c(0,log1p(max_real)),ylim=c(0,log1p(max_real)),clip="off") + theme_corr_clean()
}
pG <- make_corr("g", FALSE); pH <- make_corr("h", TRUE)

# Match the original 0502.R cowplot assembly and label treatment.
pF_main <- pF + scale_y_discrete(labels=function(x) { x <- gsub("Mosquito-Graphormer","Mosquito-\nGraphormer",x,fixed=TRUE); gsub("Graphormer combined with SEIR","Graphormer\ncombined\nwith SEIR",x,fixed=TRUE) }) + theme(plot.margin=margin(2,4,2,26), panel.spacing=unit(.35,"lines"), axis.text.y=element_text(face="bold",size=10.8,lineheight=.92), axis.text.x=element_text(size=13.2), axis.title.x=element_text(size=16), strip.text=element_text(face="bold",size=14.2))
pG_main <- pG + guides(color=guide_colourbar(direction="vertical",title.position="top",title.hjust=0,barheight=unit(3.2,"cm"),barwidth=unit(.32,"cm"))) + theme(plot.margin=margin(2,5,2,4),axis.title=element_text(size=16.2),axis.text=element_text(size=13.4),legend.position="right",legend.background=element_rect(fill="white",color=NA),legend.title=element_text(size=12.2,face="bold",lineheight=1.08),legend.text=element_text(size=11.8),legend.margin=margin(0,0,0,0),legend.box.margin=margin(0,0,0,0))
pH_main <- pH + guides(color=guide_colourbar(direction="vertical",title.position="top",title.hjust=0,barheight=unit(3.2,"cm"),barwidth=unit(.32,"cm"))) + theme(plot.margin=margin(2,18,2,4),axis.title=element_text(size=16.2),axis.text=element_text(size=13.4),legend.position="right",legend.background=element_rect(fill="white",color=NA),legend.title=element_text(size=12.2,face="bold",lineheight=1.08),legend.text=element_text(size=11.8),legend.margin=margin(0,0,0,0),legend.box.margin=margin(0,0,0,0))
row1 <- cowplot::plot_grid(pA+theme(plot.margin=margin(4,5,4,3)),pB+theme(plot.margin=margin(4,3,4,5)),labels=c("a","b"),label_size=22,label_fontfamily=font_family,label_fontface="bold",ncol=2,rel_widths=c(1.04,1),align="hv",axis="tblr")
row2 <- cowplot::plot_grid(pC+theme(plot.margin=margin(3,5,3,3)),pD+theme(plot.margin=margin(3,5,3,5)),pE+theme(plot.margin=margin(3,3,3,5)),labels=c("c","d","e"),label_size=22,label_fontfamily=font_family,label_fontface="bold",ncol=3,rel_widths=c(1,1,1),align="hv",axis="tblr")
row3_panels <- cowplot::plot_grid(pF_main,pG_main,NULL,pH_main,ncol=4,rel_widths=c(1.00,1.13,.13,1.13))
panel_letter <- function(label) cowplot::ggdraw() + cowplot::draw_label(label,x=.01,y=.5,hjust=0,fontfamily=font_family,fontface="bold",size=22)
row3_letters <- cowplot::plot_grid(panel_letter("f"),panel_letter("g"),NULL,panel_letter("h"),ncol=4,rel_widths=c(1.00,1.13,.13,1.13))
row3 <- cowplot::plot_grid(row3_letters,row3_panels,ncol=1,rel_heights=c(.07,.93))
fig <- cowplot::plot_grid(row1,row2,row3,ncol=1,rel_heights=c(1.02,1.08,1.25),align="v",axis="lr")
save_all(fig,"Figure2_V4_CoV_trend_attachment_style")
write_tsv(src,file.path(out_dir,"Figure2_UPDATED_SOURCE.tsv"))
writeLines(c("Attachment Figure2.pdf layout replot from its project R/cowplot source.","Mosquito-Graphormer values: locked V4 trend-CoV ten-seed R2; 121-city joint training.","TCN: values retained from the supplied final source table.","Graphormer combined with SEIR: values retained from the supplied final source table.","Intervals: empirical seed quantiles, not calibrated case prediction intervals."),file.path(out_dir,"README.md"))
cat(sprintf("PASS\nbackend=R\nlayout=original 0502.R cowplot\nout=%s\n", out_dir))
