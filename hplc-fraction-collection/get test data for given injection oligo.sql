USE Purification;

DECLARE @mfgItemId INT = 813432813;

SELECT DISTINCT
       injectionOligo.MfgItemId AS [InjectionOligoMfgItemId],
       --spawnOligo.MfgItemId AS [SpawnOligoMfgItemId],
       hrd.MethodId,
       specOligo.PROD_ID AS [ProductId],
       specOligo.GUAR_MIN_SHIP_VAL * 1E6 / ssi.EXT_COEFF AS [NanomoleGuarantee],
       specOligo.GUAR_MIN_SHIP_VAL AS [ShipODGuarantee],
       CASE
           WHEN EXISTS
                (
                    SELECT 1
                    FROM Reference.dbo.SCI_MODS AS postSynthTaggedSciMod
                    WHERE postSynthTaggedSciMod.PROD_ID = ssp.PROD_ID
                          AND postSynthTaggedSciMod.POST_SYNTH_TAGGED = 1
                ) THEN
               1
           ELSE
               0
       END AS [IsToBeTagged],
       f.VialPosition,
       CASE
           WHEN f.SpawnOligoId IS NOT NULL THEN
               1
           ELSE
               0
       END AS [VialWasSelected]
FROM dbo.Oligo AS injectionOligo
    JOIN dbo.Fraction AS f
        ON f.InjectionOligoId = injectionOligo.OligoId
    --LEFT JOIN dbo.Oligo AS spawnOligo
    --    ON spawnOligo.OligoId = f.SpawnOligoId
    JOIN dbo.HPLCRunDetail AS hrd
        ON hrd.HPLCRunDetailId = f.HPLCRunDetailId
    JOIN Production.dbo.SPEC_OLIGO AS specOligo
        ON injectionOligo.SpecOligoId = specOligo.SPEC_OLIGO_ID
    LEFT JOIN Production.dbo.SPEC_SEQ_INFO AS ssi
        ON ssi.SPEC_OLIGO_ID = specOligo.SPEC_OLIGO_ID
    LEFT JOIN Production.dbo.SPEC_SEQ_PRODUCT AS ssp
        ON ssp.SPEC_OLIGO_ID = specOligo.SPEC_OLIGO_ID
WHERE injectionOligo.MfgItemId = @mfgItemId
ORDER BY f.VialPosition;