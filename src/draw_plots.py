## TBA: Generate box plots from csv files

# def run_complete_expe(expe_name,ds,label_list,weighted=False):
#     ds = ds.remove_columns("label")
#     print('====')
#     print(expe_name)
#     print('====')
#     print('running conversation ')
#     res_cv_conv = run_crossvalid(expe_name,ds,'../toy_llm/models/fr_10M_10K_conv/',label_list,
#                                  weighted=weighted,verbose=False,error_analysis=True,keep_models=False)
#     print('running wiki')
#     res_cv_wiki = run_crossvalid(expe_name,ds,'../toy_llm/models/fr_10M_10K_wiki/',label_list,
#                                  weighted=weighted,verbose=False,error_analysis=True,keep_models=False)
#     print('running roberta')
#     res_cv_rob = run_crossvalid(expe_name,ds,'xlm-roberta-base',label_list,
#                                 weighted=weighted,verbose=False,error_analysis=True,keep_models=False)

#     res_cv_conv_df = pd.DataFrame(res_cv_conv)
#     res_cv_wiki_df = pd.DataFrame(res_cv_wiki)
#     res_cv_rob_df = pd.DataFrame(res_cv_rob)

#     res_cv_conv_df['model'] = 'conv10K'
#     res_cv_wiki_df['model'] = 'wiki10K'
#     res_cv_rob_df['model'] = 'roberta'

#     res_cv_all = pd.concat([res_cv_conv_df,res_cv_wiki_df,res_cv_rob_df])

#     res_cv_all.to_csv(RESULT_FOLDER+expe_name+'10K_cv.csv')

#     plot = sns.boxplot(data=res_cv_all,x='model',y='fs')
#     plot.figure.savefig(FIGS_FOLDER+expe_name+'_fscore.png',dpi=300)
#     plot.figure.clf() 

#     plot = sns.boxplot(data=res_cv_all,x='model',y='prec')
#     plot.figure.savefig(FIGS_FOLDER+expe_name+'prec.png',dpi=300)
#     plot.figure.clf() 

#     plot = sns.boxplot(data=res_cv_all,x='model',y='rec')
#     plot.figure.savefig(FIGS_FOLDER+expe_name+'rec.png',dpi=300)
#     plot.figure.clf() 
#     return 0