# commands

## command to check images still waiting for processing

`az storage blob list \
  --account-name reprlearnnishantdev \
  --container-name raw-images \
  --auth-mode login \
  --query "[].{name:name, size:properties.contentLength}" \
  --output table`


## command to check accepted image
`az storage blob list \
  --account-name reprlearnnishantdev \
  --container-name accepted-images \
  --auth-mode login \
  --query "[].{name:name, size:properties.contentLength}" \
  --output table`

## command to check quarentined image
`az storage blob list \
  --account-name reprlearnnishantdev \
  --container-name quarantined-images \
  --auth-mode login \
  --query "[].{name:name, size:properties.contentLength}" \
  --output table
`

## command to check service bus queue
`az servicebus queue show \
  --name image-ingestion \
  --namespace-name sb-reprlearn-nishant-dev \
  --resource-group rg-representation-learning-dev \
  --query "countDetails.{active:activeMessageCount, deadLetter:deadLetterMessageCount}" \
  --output table
`

## git command to find occurences of a text
`git grep -n "BlobCreatedEventHandler("
`
