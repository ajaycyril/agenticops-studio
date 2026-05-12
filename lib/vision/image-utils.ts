export function estimateImageMetadata(imageBase64?: string) {
  if (!imageBase64) return { imageSizeBytes: 0, imageType: "sample" };
  const commaIndex = imageBase64.indexOf(",");
  const header = commaIndex >= 0 ? imageBase64.slice(0, commaIndex) : "";
  const payload = commaIndex >= 0 ? imageBase64.slice(commaIndex + 1) : imageBase64;
  const imageType = header.includes("image/") ? header.split("image/")[1]?.split(";")[0] ?? "unknown" : "unknown";
  return {
    imageSizeBytes: Math.round((payload.length * 3) / 4),
    imageType
  };
}

export function stripDataUrl(imageBase64: string) {
  const commaIndex = imageBase64.indexOf(",");
  return commaIndex >= 0 ? imageBase64.slice(commaIndex + 1) : imageBase64;
}
