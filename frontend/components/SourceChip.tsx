import styles from "./SourceChip.module.css";

interface Props { source: string; }

export default function SourceChip({ source }: Props) {
  const isUrl = source.startsWith("http");
  const label = isUrl ? new URL(source).hostname : source;

  return isUrl ? (
    <a href={source} target="_blank" rel="noopener noreferrer" className={styles.chip}>
      <span className={styles.dot} />
      {label}
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" />
        <polyline points="15 3 21 3 21 9" /><line x1="10" y1="14" x2="21" y2="3" />
      </svg>
    </a>
  ) : (
    <span className={styles.chip}>
      <span className={styles.dot} />
      {label}
    </span>
  );
}
